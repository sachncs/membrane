"""Tests for PayloadIdentity (the new ten-field fragment fingerprint)."""

from membrane.identity import PayloadIdentity


def test_create_payload_identity():
    ident = PayloadIdentity(
        payload_hash="a" * 64,
        model_id="kimi-linear-1t",
        model_revision="",
        tokenizer_name="kimi-linear-1t",
        tokenizer_revision="",
        layer_range=(0, 3),
        head_range=(-1, -1),
        token_span=(1024, 2048),
        dtype="float16",
        shape=(1, 1, 1, 1024, 64),
    )
    assert ident.model_id == "kimi-linear-1t"
    assert ident.layer_range == (0, 3)
    assert ident.token_span == (1024, 2048)
    assert ident.head_range == (-1, -1)
    assert ident.shape == (1, 1, 1, 1024, 64)


def test_payload_identity_is_hashable():
    ident = PayloadIdentity(
        payload_hash="a",
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 10),
        dtype="float16",
        shape=(1, 1, 1, 11, 64),
    )
    assert hash(ident) == hash(ident)


def test_payload_identity_equality():
    a = PayloadIdentity(
        payload_hash="h",
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 10),
        dtype="float16",
        shape=(1, 1, 1, 11, 64),
    )
    b = PayloadIdentity(
        payload_hash="h",
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 10),
        dtype="float16",
        shape=(1, 1, 1, 11, 64),
    )
    c = PayloadIdentity(
        payload_hash="h",
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 2),
        head_range=(-1, -1),
        token_span=(0, 10),
        dtype="float16",
        shape=(1, 1, 1, 11, 64),
    )
    assert a == b
    assert a != c


def test_fingerprint_is_deterministic():
    ident = PayloadIdentity(
        payload_hash="h",
        model_id="m",
        model_revision="r1",
        tokenizer_name="m",
        tokenizer_revision="r2",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 10),
        dtype="float16",
        shape=(1, 1, 1, 11, 64),
    )
    assert ident.fingerprint() == ident.fingerprint()
