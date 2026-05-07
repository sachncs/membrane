import pytest

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


def test_create_fragment():
    sig = StructuralSignature("m", (0, 3), (0, 1024))
    frag = Fragment(
        content_hash="abc123",
        embedding=(0.1, 0.2, 0.3),
        structural_signature=sig,
        size=1024,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )
    assert frag.content_hash == "abc123"
    assert frag.size == 1024
    assert frag.version_id == 1
    assert isinstance(frag.embedding, tuple)


def test_fragment_is_immutable():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h", (0.1,), sig, 10, 60.0, 0.5, 1)
    with pytest.raises(AttributeError):
        frag.size = 20


def test_fragment_is_hashable():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h", (0.1,), sig, 10, 60.0, 0.5, 1)
    assert hash(frag) == hash(("h", (0.1,), sig, 10, 60.0, 0.5, 1))
    assert len({frag}) == 1


def test_fragment_equality():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    a = Fragment("h", (0.1,), sig, 10, 60.0, 0.5, 1)
    b = Fragment("h", (0.1,), sig, 10, 60.0, 0.5, 1)
    c = Fragment("h2", (0.1,), sig, 10, 60.0, 0.5, 1)
    assert a == b
    assert a != c


# --- Validation tests designed to break invalid construction ---


def test_negative_size_rejected():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    with pytest.raises(ValueError, match="size must be >= 0"):
        Fragment("h", (0.1,), sig, -1, 60.0, 0.5, 1)


def test_negative_ttl_rejected():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    with pytest.raises(ValueError, match="ttl must be >= 0"):
        Fragment("h", (0.1,), sig, 10, -1.0, 0.5, 1)


def test_reuse_score_below_zero_rejected():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    with pytest.raises(ValueError, match=r"reuse_score must be in \[0, 1\]"):
        Fragment("h", (0.1,), sig, 10, 60.0, -0.1, 1)


def test_reuse_score_above_one_rejected():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    with pytest.raises(ValueError, match=r"reuse_score must be in \[0, 1\]"):
        Fragment("h", (0.1,), sig, 10, 60.0, 1.1, 1)


def test_version_id_zero_rejected():
    sig = StructuralSignature("m", (0, 1), (0, 10))
    with pytest.raises(ValueError, match="version_id must be >= 1"):
        Fragment("h", (0.1,), sig, 10, 60.0, 0.5, 0)


def test_boundary_values_accepted():
    """Boundary values (0, 1.0) should be valid."""
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h", (0.1,), sig, 0, 0.0, 0.0, 1)
    assert frag.size == 0
    assert frag.ttl == 0.0
    assert frag.reuse_score == 0.0
    frag2 = Fragment("h", (0.1,), sig, 0, 0.0, 1.0, 1)
    assert frag2.reuse_score == 1.0
