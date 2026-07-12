"""Tests for kv_cache_manager module."""

import pytest

from membrane.cache_metrics import CacheMetrics
from membrane.fragment import Fragment
from membrane.fragmentation_engine import compute_content_hash
from membrane.index_system import IndexSystem
from membrane.kv_cache_manager import KVCacheManager
from membrane.structural_signature import StructuralSignature


def make_fragment(token_span=(0, 3), content_hash="abc"):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2),
        structural_signature=StructuralSignature(model_id="test-model", layer_range=(0, 1), token_span=token_span),
        size=128,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestKVCacheManager:
    """Test suite for KVCacheManager."""

    def test_lookup_miss_empty_index(self):
        mgr = KVCacheManager()
        result = mgr.lookup_kv("missing")
        assert result == []
        assert mgr.get_miss_rate() == 1.0

    def test_store_and_lookup_hit_by_prefix(self):
        """Prefix lookup should return fragments stored under that prefix."""
        mgr = KVCacheManager()
        frag = make_fragment(content_hash="hit-hash")
        mgr.store_kv("prefix-a", [frag])
        result = mgr.lookup_kv("prefix-a")
        assert len(result) == 1
        assert result[0].content_hash == "hit-hash"
        assert mgr.get_hit_rate() == 1.0

    def test_lookup_byfragment_hash_is_miss(self):
        """Looking up by fragment content_hash should be a miss;
        the manager keys by prefix_hash, not fragment hash."""
        mgr = KVCacheManager()
        frag = make_fragment(content_hash="hit-hash")
        mgr.store_kv("prefix-a", [frag])
        result = mgr.lookup_kv("hit-hash")
        assert result == []
        assert mgr.get_miss_rate() == 1.0

    def test_multiplefragments_under_same_prefix(self):
        mgr = KVCacheManager()
        f1 = make_fragment(content_hash="f1")
        f2 = make_fragment(content_hash="f2")
        mgr.store_kv("prefix-a", [f1, f2])
        result = mgr.lookup_kv("prefix-a")
        assert len(result) == 2
        hashes = {f.content_hash for f in result}
        assert hashes == {"f1", "f2"}

    def test_store_overwrites_prefix_mapping(self):
        """Storing again under the same prefix should replace old fragments."""
        mgr = KVCacheManager()
        f1 = make_fragment(content_hash="f1")
        f2 = make_fragment(content_hash="f2")
        mgr.store_kv("prefix-a", [f1])
        mgr.store_kv("prefix-a", [f2])
        result = mgr.lookup_kv("prefix-a")
        assert len(result) == 1
        assert result[0].content_hash == "f2"

    def test_remove_prefix(self):
        mgr = KVCacheManager()
        frag = make_fragment(content_hash="f1")
        mgr.store_kv("prefix-a", [frag])
        assert mgr.remove_prefix("prefix-a") is True
        assert mgr.lookup_kv("prefix-a") == []
        assert mgr.remove_prefix("prefix-a") is False

    def test_metrics_snapshot(self):
        mgr = KVCacheManager()
        mgr.lookup_kv("x")
        mgr.lookup_kv("y")
        metrics = mgr.get_metrics()
        assert isinstance(metrics, CacheMetrics)
        assert metrics.misses == 2
        assert metrics.total_requests == 2

    def test_store_with_custom_node_id(self):
        mgr = KVCacheManager()
        frag = make_fragment(content_hash="custom")
        mgr.store_kv("p", [frag], node_id="node-7")
        entry = mgr.index_system.exact_lookup("custom")
        assert entry is not None
        assert "node-7" in entry.locations

    def test_hit_rate_after_mixed_access(self):
        mgr = KVCacheManager()
        frag = make_fragment(content_hash="known")
        mgr.store_kv("p", [frag])
        mgr.lookup_kv("p")
        mgr.lookup_kv("unknown")
        assert mgr.get_hit_rate() == 0.5
        assert mgr.get_miss_rate() == 0.5

    def test_uses_provided_index_system(self):
        idx = IndexSystem()
        mgr = KVCacheManager(index_system=idx)
        frag = make_fragment(content_hash="shared")
        mgr.store_kv("p", [frag])
        assert idx.exact_lookup("shared") is not None

    def test_lookup_returns_empty_list_not_none_on_miss(self):
        """lookup_kv must always return a list, never None."""
        mgr = KVCacheManager()
        result = mgr.lookup_kv("nonexistent")
        assert result == []
        assert isinstance(result, list)

    def test_lru_eviction_on_overflow(self):
        mgr = KVCacheManager(max_prefixes=2)
        mgr.store_kv("p1", [make_fragment(content_hash="a")])
        mgr.store_kv("p2", [make_fragment(content_hash="b")])
        mgr.store_kv("p3", [make_fragment(content_hash="c")])
        assert len(mgr.prefix_to_fragments) == 2
        assert "p1" not in mgr.prefix_to_fragments

    def test_lru_keeps_recently_accessed_prefix(self):
        mgr = KVCacheManager(max_prefixes=2)
        mgr.store_kv("p1", [make_fragment(content_hash="a")])
        mgr.store_kv("p2", [make_fragment(content_hash="b")])
        mgr.lookup_kv("p1")  # make p1 recently used
        mgr.store_kv("p3", [make_fragment(content_hash="c")])
        assert "p1" in mgr.prefix_to_fragments
        assert "p2" not in mgr.prefix_to_fragments
