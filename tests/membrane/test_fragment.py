import dataclasses

import pytest

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


def _identity(payload_hash: str = "abc123", model_id: str = "m", token_span: tuple[int, int] = (0, 1024)) -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash=payload_hash,
        model_id=model_id,
        model_revision="",
        tokenizer_name=model_id,
        tokenizer_revision="",
        layer_range=(0, 3),
        head_range=(-1, -1),
        token_span=token_span,
        dtype="float16",
        shape=(1, 4, 4, 128, 64),
    )


def test_create_fragment():
    ident = _identity()
    frag = Fragment(
        identity=ident,
        payload_ref="abc123",
        payload_size=1024,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )
    assert frag.identity.payload_hash == "abc123"
    assert frag.payload_size == 1024
    assert frag.version_id == 1


def test_fragment_is_immutable():
    ident = _identity("h", token_span=(0, 10))
    frag = Fragment(
        identity=ident,
        payload_ref="h",
        payload_size=10,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
    )
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        frag.payload_size = 20  # type: ignore[misc]


def test_fragment_is_hashable():
    import dataclasses
    ident = _identity("h", token_span=(0, 10))
    frag = Fragment(
        identity=ident,
        payload_ref="h",
        payload_size=10,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
    )
    assert hash(frag) == hash(
    (ident, "h", 10, 60.0, 0.5, 1, "strong", 0, "", "public")
)
    assert len({frag}) == 1


def test_fragment_equality():
    ident = _identity("h", token_span=(0, 10))
    a = Fragment(identity=ident, payload_ref="h", payload_size=10, ttl=60.0, reuse_score=0.5, version_id=1)
    b = Fragment(identity=ident, payload_ref="h", payload_size=10, ttl=60.0, reuse_score=0.5, version_id=1)
    c_ident = _identity("h2", token_span=(0, 10))
    c = Fragment(identity=c_ident, payload_ref="h2", payload_size=10, ttl=60.0, reuse_score=0.5, version_id=1)
    assert a == b
    assert a != c


# --- Validation tests designed to break invalid construction ---


def test_negative_payload_size_rejected():
    ident = _identity("h", token_span=(0, 10))
    with pytest.raises(ValueError, match="payload_size must be >= 0"):
        Fragment(identity=ident, payload_ref="h", payload_size=-1, ttl=60.0, reuse_score=0.5, version_id=1)


def test_negative_ttl_rejected():
    ident = _identity("h", token_span=(0, 10))
    with pytest.raises(ValueError, match="ttl must be >= 0"):
        Fragment(identity=ident, payload_ref="h", payload_size=10, ttl=-1.0, reuse_score=0.5, version_id=1)


def test_reuse_score_below_zero_rejected():
    ident = _identity("h", token_span=(0, 10))
    with pytest.raises(ValueError, match=r"reuse_score must be in \[0, 1\]"):
        Fragment(identity=ident, payload_ref="h", payload_size=10, ttl=60.0, reuse_score=-0.1, version_id=1)


def test_reuse_score_above_one_rejected():
    ident = _identity("h", token_span=(0, 10))
    with pytest.raises(ValueError, match=r"reuse_score must be in \[0, 1\]"):
        Fragment(identity=ident, payload_ref="h", payload_size=10, ttl=60.0, reuse_score=1.1, version_id=1)


def test_version_id_zero_rejected():
    ident = _identity("h", token_span=(0, 10))
    with pytest.raises(ValueError, match="version_id must be >= 1"):
        Fragment(identity=ident, payload_ref="h", payload_size=10, ttl=60.0, reuse_score=0.5, version_id=0)


def test_boundary_values_accepted():
    """Boundary values (0, 1.0) should be valid."""
    ident = _identity("h", token_span=(0, 10))
    frag = Fragment(identity=ident, payload_ref="h", payload_size=0, ttl=0.0, reuse_score=0.0, version_id=1)
    assert frag.payload_size == 0
    assert frag.ttl == 0.0
    assert frag.reuse_score == 0.0
    ident2 = _identity("h", token_span=(0, 10))
    frag2 = Fragment(identity=ident2, payload_ref="h", payload_size=0, ttl=0.0, reuse_score=1.0, version_id=1)
    assert frag2.reuse_score == 1.0


def test_identity_validates():
    with pytest.raises(ValueError):
        PayloadIdentity(
            payload_hash="x",
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 0),
            head_range=(0, 0),
            token_span=(1, 0),
            dtype="float16",
            shape=(1,),
        )
