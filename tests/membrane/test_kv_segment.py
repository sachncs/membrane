"""Tests for KVSegment memory object."""

from membrane.fragmentation_engine import compute_content_hash
from membrane.kv_segment import KVSegment


def test_kv_segment_creation():
    seg = KVSegment(
        layer=3,
        head=7,
        token_span=(0, 127),
        tensor_shape=(32, 128, 128),
        content_hash=compute_content_hash((3, 7, 0, 127)),
        semantic_hash="sh",
        size_bytes=1024,
        reuse_score=0.9,
    )
    assert seg.layer == 3
    assert seg.head == 7
    assert seg.size_bytes == 1024


def test_kv_segment_materialize():
    seg = KVSegment(
        layer=1,
        head=0,
        token_span=(0, 63),
        tensor_shape=(1, 64, 64),
        content_hash=compute_content_hash((1, 0, 0, 63)),
        semantic_hash="sh",
        size_bytes=512,
        reuse_score=0.5,
    )
    frag = seg.materialize()
    assert frag.content_hash == seg.content_hash
    assert frag.structural_signature.layer_range == (1, 1)


def test_kv_segment_from_fragment():
    seg = KVSegment(
        layer=2,
        head=4,
        token_span=(10, 20),
        tensor_shape=(1, 1, 1),
        content_hash="h123",
        semantic_hash="sh",
        size_bytes=128,
        reuse_score=0.3,
    )
    frag = seg.materialize()
    recon = KVSegment.from_fragment(frag)
    assert recon.layer == 2
    assert recon.size_bytes == 128
