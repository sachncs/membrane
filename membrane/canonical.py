"""Canonical byte framing for fragment payloads.

A "canonical" frame is the immutable on-disk / on-the-wire representation
of a single fragment payload. Frames are self-describing: the header
carries the :class:`~membrane.identity.PayloadIdentity` and a payload
length, and the trailer carries a truncated SHA-256 for cheap integrity
checks on read without re-parsing the entire blob.

Frame layout (schema v5)::

    +-------------------------------+
    | MAGIC       4 B  = 0xC0DE0105 |   (last byte = 0x05 for v5)
    +-------------------------------+
    | schema      2 B              |   (= 5 for v5; the on-disk
    +-------------------------------+    version is the wire schema)
    | reserved    4 B  (= 0)        |
    +-------------------------------+
    | identity_len u32 (LE)         |
    +-------------------------------+   offset = 14
    | identity_json (identity_len B)|   (UTF-8 JSON of
    +-------------------------------+    PayloadIdentity.to_dict())
    | payload_len  u64 (LE)         |
    +-------------------------------+
    | payload      (payload_len B)  |
    +-------------------------------+
    | trailer      8 B              |   (first 8 bytes of SHA-256
    +-------------------------------+    of payload; cheap verify)

The trailer is verified in :func:`parse_canonical`. A mismatch raises
:class:`membrane.errors.CorruptPayloadError` rather than retry, because a
mismatch indicates storage corruption, not transient failure.

The v3.0.0 release accepts only the v5 magic. Older versions (v2, v4)
are hard-failed with :class:`membrane.errors.SchemaError`; there is no
backward-compatible reader. Operators upgrading from a 2.0 deployment
must convert frames via a one-shot migration script before booting a
3.0.0 cluster.

Frames are immutable; the on-disk file should be written atomically
(temp file + :func:`os.replace`) by the storage backend, not by this
module.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Final

from membrane.errors import CorruptPayloadError, SchemaError
from membrane.identity import PayloadIdentity

MAGIC: Final[bytes] = b"\xc0\xde\x01\x05"
HEADER_LEN: Final[int] = 14
TRAILER_LEN: Final[int] = 8
CANONICAL_SCHEMA_VERSION: Final[int] = 5


def canonicalize(identity: PayloadIdentity, raw: bytes) -> bytes:
    """Wrap an identity + payload into the canonical frame.

    Args:
        identity: The fragment's :class:`PayloadIdentity`.
        raw: The raw payload bytes (e.g. serialized tensor bytes).

    Returns:
        bytes: The complete frame, ready to write to any blob backend.

    Raises:
        ValueError: If ``raw`` exceeds the 64-bit length limit.
    """
    if len(raw) > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"payload too large to frame: {len(raw)} bytes")
    identity_bytes = json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    header = MAGIC + struct.pack("<HI", CANONICAL_SCHEMA_VERSION, 0) + struct.pack("<I", len(identity_bytes))
    body = identity_bytes + struct.pack("<Q", len(raw)) + raw
    digest = hashlib.sha256(raw).digest()[:TRAILER_LEN]
    return header + body + digest


def parse_canonical(buf: bytes) -> tuple[PayloadIdentity, bytes]:
    """Parse a canonical frame back into identity + payload.

    The v3.0.0 reader accepts only v5 frames. Older schemas
    raise :class:`~membrane.errors.SchemaError` with a clear
    message; operators upgrading from 2.x must run the
    one-shot migration script documented in CHANGELOG.

    Args:
        buf: The full frame produced by :func:`canonicalize`.

    Returns:
        tuple[PayloadIdentity, bytes]: The fingerprint and the raw
        payload bytes.

    Raises:
        SchemaError: If the magic or schema version does not
            match v5. There is no backward-compatible reader.
        CorruptPayloadError: If the trailer's truncated SHA-256
            disagrees with the payload bytes.
    """
    if len(buf) < HEADER_LEN + TRAILER_LEN:
        raise CorruptPayloadError(f"frame too short: {len(buf)} bytes")
    if buf[:4] != MAGIC:
        raise SchemaError(f"bad magic in canonical frame: {buf[:4]!r}")
    schema = struct.unpack_from("<H", buf, 4)[0]
    if schema != CANONICAL_SCHEMA_VERSION:
        raise SchemaError(
            f"canonical schema version mismatch: {schema} vs {CANONICAL_SCHEMA_VERSION}"
        )
    identity_len = struct.unpack_from("<I", buf, 10)[0]
    identity_end = HEADER_LEN + identity_len
    if identity_end + 8 > len(buf):
        raise CorruptPayloadError("identity length extends past frame")
    identity_obj = json.loads(buf[HEADER_LEN:identity_end].decode("utf-8"))
    identity = PayloadIdentity.from_dict(identity_obj)
    payload_len = struct.unpack_from("<Q", buf, identity_end)[0]
    payload_start = identity_end + 8
    payload_end = payload_start + payload_len
    if payload_end + TRAILER_LEN > len(buf):
        raise CorruptPayloadError("payload length extends past frame")
    payload = bytes(buf[payload_start:payload_end])
    expected = hashlib.sha256(payload).digest()[:TRAILER_LEN]
    actual = bytes(buf[payload_end:payload_end + TRAILER_LEN])
    if expected != actual:
        raise CorruptPayloadError("canonical frame trailer mismatch")
    return identity, payload


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "HEADER_LEN",
    "MAGIC",
    "TRAILER_LEN",
    "canonicalize",
    "parse_canonical",
]
