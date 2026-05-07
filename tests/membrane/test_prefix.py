"""Tests for Prefix memory object."""

from membrane.delta_encoder import Delta
from membrane.fragmentation_engine import compute_content_hash
from membrane.prefix import Prefix
from membrane.semantic_hash import compute_semantic_hash


def test_prefix_creation():
    tokens = (1, 2, 3, 4)
    p = Prefix(
        tokens=tokens,
        content_hash=compute_content_hash(tokens),
        semantic_hash=compute_semantic_hash((0.1, 0.2)),
        size_bytes=256,
        token_count=4,
        reuse_score=0.8,
    )
    assert p.token_count == 4
    assert p.reuse_score == 0.8


def test_prefix_materialize():
    tokens = (10, 20, 30)
    p = Prefix(
        tokens=tokens,
        content_hash=compute_content_hash(tokens),
        semantic_hash=compute_semantic_hash((0.1, 0.2)),
        size_bytes=128,
        token_count=3,
        reuse_score=0.5,
    )
    frag = p.materialize()
    assert frag.content_hash == p.content_hash
    assert frag.structural_signature.token_span == (0, 2)


def test_prefix_from_fragment():
    tokens = (1, 2)
    p = Prefix(
        tokens=tokens,
        content_hash=compute_content_hash(tokens),
        semantic_hash="abc",
        size_bytes=64,
        token_count=2,
        reuse_score=0.5,
    )
    frag = p.materialize()
    reconstructed = Prefix.from_fragment(frag)
    assert reconstructed.content_hash == p.content_hash
    assert reconstructed.token_count == p.token_count


def test_prefix_delta_creation():
    delta = Delta(
        base_content_hash="base123",
        appended_tokens=(5, 6, 7),
        removed_tail_count=0,
    )
    assert delta.base_content_hash == "base123"
    assert delta.appended_tokens == (5, 6, 7)


def test_prefix_is_addressable():
    tokens = (42,)
    p = Prefix(
        tokens=tokens,
        content_hash=compute_content_hash(tokens),
        semantic_hash="sh",
        size_bytes=32,
        token_count=1,
        reuse_score=0.1,
    )
    assert isinstance(p.content_hash, str)
    assert isinstance(p.semantic_hash, str)
