"""Tests for MembraneNode."""

import time

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature


def make_fragment(
    content_hash: str, size: int, reuse_score: float = 0.5, ttl: float = 3600.0
) -> Fragment:
    sig = StructuralSignature("m", (0, 1), (0, 10))
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2, 0.3),
        structural_signature=sig,
        size=size,
        ttl=ttl,
        reuse_score=reuse_score,
        version_id=1,
    )


def test_store_increases_memory():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    frag = make_fragment("h1", 1000)
    assert node.store(frag)
    assert node.get_memory_usage() == 1000


def test_retrieve_correct_fragment():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    frag = make_fragment("h1", 1000)
    node.store(frag)
    result = node.retrieve("h1")
    assert result == frag


def test_retrieve_missing_returns_none():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    assert node.retrieve("missing") is None


def test_evict_respects_max_memory():
    node = MembraneNode("n1", max_memory_bytes=500)
    a = make_fragment("a", 200, reuse_score=0.1)
    b = make_fragment("b", 200, reuse_score=0.9)
    c = make_fragment("c", 200, reuse_score=0.5)
    node.store(a)
    node.store(b)
    node.store(c)
    assert node.get_memory_usage() <= 500


def test_ttl_expiry_evicts_first():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    expired = make_fragment("old", 100, ttl=0.01)
    fresh = make_fragment("new", 100, ttl=3600.0)
    node.store(expired)
    node.store(fresh)
    time.sleep(0.02)
    evicted = node.evict(50, current_time=time.time())
    assert "old" in evicted
    assert "new" not in evicted


def test_graph_aware_eviction():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    a = make_fragment("a", 100, reuse_score=0.1)
    b = make_fragment("b", 100, reuse_score=0.1)
    node.store(a)
    node.store(b)
    node.graph_manager.link("a", "b", "co_access")
    node.evict(50)
    # a is evicted because low reuse_score, b may follow via graph-aware
    assert "a" not in node.fragments or "b" not in node.fragments


def test_store_rejects_too_large():
    node = MembraneNode("n1", max_memory_bytes=100)
    frag = make_fragment("big", 200)
    assert not node.store(frag)


def test_shard_ownership_tracked():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    frag = make_fragment("h1", 100)
    node.store(frag, is_primary=True)
    assert node.get_shard_hashes() == {"h1"}
    node.store(frag, is_primary=False)
    assert node.get_shard_hashes() == {"h1"}


def test_heartbeat_empty_node():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    assert node.heartbeat() == 0.0


def test_mixed_stores_and_evictions():
    node = MembraneNode("n1", max_memory_bytes=300)
    frags = [make_fragment(f"h{i}", 100, reuse_score=0.5 / (i + 1)) for i in range(5)]
    for f in frags:
        node.store(f)
    assert node.get_memory_usage() <= 300
    assert len(node.fragments) <= 3


def test_retrieve_evicts_expired_fragment():
    """Background TTL expiry: retrieve should evict expired fragments."""
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    frag = make_fragment("old", 100, ttl=0.01)
    node.store(frag)
    time.sleep(0.02)
    result = node.retrieve("old")
    assert result is None
    assert "old" not in node.fragments


def test_retrieve_does_not_evict_fresh_fragment():
    node = MembraneNode("n1", max_memory_bytes=1_000_000)
    frag = make_fragment("fresh", 100, ttl=3600.0)
    node.store(frag)
    result = node.retrieve("fresh")
    assert result == frag
    assert "fresh" in node.fragments
