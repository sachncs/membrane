"""Canonical byte framing for fragment payloads.

A "canonical" frame is the immutable on-disk / on-the-wire representation
of a single fragment payload. Frames are self-describing: the header
carries the :class:`~membrane.identity.PayloadIdentity` and a payload
length, and the trailer carries a truncated SHA-256 for cheap integrity
checks on read without re-parsing the entire blob.

Frame layout (schema v4)::

    +-------------------------------+
    | MAGIC       4 B  = 0xC0DE0104 |   (last byte = 0x04 for v4)
    +-------------------------------+
    | schema      2 B              |   (= 4 for v4; the on-disk
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

The frame's binary layout is intentionally compatible with
LMCache's ``MemoryObj`` body: ``identity_json`` is the
:class:`~membrane.identity.PayloadIdentity` payload (the model
and tokenizer metadata), and ``payload`` is the same byte
stream LMCache stores inside a ``TensorMemoryObj`` once the
canonical frame has been packed. LMCache's high-level engine
treating the canonical frame as a K/V tensor body is wired in
Phase 5+; the v1 of this module just keeps the on-disk layout
identical so a future LMCache ``connector`` can read it without
extra translation.

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

MAGIC: Final[bytes] = b"\xc0\xde\x01\x04"
#: Fixed header length = MAGIC (4) + schema (2) + reserved (4) + identity_len (4)
HEADER_LEN: Final[int] = 14
TRAILER_LEN: Final[int] = 8
CANONICAL_SCHEMA_VERSION: Final[int] = 4
#: v2 schema magic for backward-compatible reading. The v4 reader
#: also accepts v2 frames so existing on-disk blobs survive the
#: upgrade; the v2 writer is no longer available.
_CANONICAL_V2_MAGIC: Final[bytes] = b"\xc0\xde\x01\x02"
_CANONICAL_V2_SCHEMA: Final[int] = 2


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

    The v4 reader accepts v2 frames (CANONICAL_SCHEMA_VERSION=2) so
    on-disk blobs written by the 1.x series survive the upgrade.
    The 2.0 writer only produces v4 frames; old readers are
    deliberately not supported.

    Args:
        buf: The full frame produced by :func:`canonicalize`.

    Returns:
        tuple[PayloadIdentity, bytes]: The fingerprint and the raw
        payload bytes.

    Raises:
        SchemaError: If the magic or schema version does not match.
        CorruptPayloadError: If the trailer's truncated SHA-256 disagrees
            with the payload bytes.
    """
    if len(buf) < HEADER_LEN + TRAILER_LEN:
        raise CorruptPayloadError(f"frame too short: {len(buf)} bytes")
    if buf[:4] != MAGIC and buf[:4] != _CANONICAL_V2_MAGIC:
        raise SchemaError(f"bad magic in canonical frame: {buf[:4]!r}")
    schema = struct.unpack_from("<H", buf, 4)[0]
    if schema not in (CANONICAL_SCHEMA_VERSION, _CANONICAL_V2_SCHEMA):
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
