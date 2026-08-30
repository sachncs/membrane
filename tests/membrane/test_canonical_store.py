"""Tests for canonical byte framing (canonical module)."""

from membrane.canonical import (
    CANONICAL_SCHEMA_VERSION,
    MAGIC,
    canonicalize,
    parse_canonical,
)
from membrane.identity import PayloadIdentity


def _identity() -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash="a" * 64,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 7),
        dtype="float16",
        shape=(1, 1, 1, 8, 64),
    )


def test_canonicalize_round_trip():
    identity = _identity()
    payload = b"hello, world"
    buf = canonicalize(identity, payload)
    parsed_identity, parsed_payload = parse_canonical(buf)
    assert parsed_identity == identity
    assert parsed_payload == payload


def test_canonicalize_schema_version_constant():
    # v3.0.0 reader accepts only v5 frames. The constant is
    # bumped in lockstep with the on-disk layout; there is no
    # backward-compatible reader.
    assert CANONICAL_SCHEMA_VERSION == 5
    assert MAGIC == b"\xc0\xde\x01\x05"


def test_parse_canonical_rejects_bad_magic():
    from membrane.errors import SchemaError

    identity = _identity()
    buf = canonicalize(identity, b"x")
    # Corrupt the magic header
    bad = b"\x00\x00\x00\x00" + buf[4:]
    try:
        parse_canonical(bad)
    except SchemaError:
        pass
    else:
        raise AssertionError("expected SchemaError on bad magic")


def test_parse_canonical_rejects_truncated_frame():
    from membrane.errors import CorruptPayloadError

    identity = _identity()
    buf = canonicalize(identity, b"abc")
    try:
        parse_canonical(buf[:5])
    except CorruptPayloadError:
        pass
    else:
        raise AssertionError("expected CorruptPayloadError on truncated frame")
