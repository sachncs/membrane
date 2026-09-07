from tests.conftest import make_fragment

"""Tests for kv_cache_manager module."""

import pytest

from membrane.cache_metrics import CacheMetrics
from membrane.fragment import Fragment
from membrane.fragmenter import compute_content_hash
from membrane.index import Index
from membrane.kv import KVCache


class TestKVCacheManager:
    """Test suite for KVCache."""

    def test_lookup_miss_empty_index(self):
        mgr = KVCache()
        result = mgr.lookup_kv("missing")
        assert result == []
        assert mgr.get_miss_rate() == 1.0

    def test_store_and_lookup_hit_by_prefix(self):
        """Prefix lookup should return fragments stored under that prefix."""
        mgr = KVCache()
        frag = make_fragment(content_hash="hit-hash")
        mgr.store_kv("prefix-a", [frag])
        result = mgr.lookup_kv("prefix-a")
        assert len(result) == 1
        assert result[0].identity.payload_hash == "hit-hash"
        assert mgr.get_hit_rate() == 1.0

    def test_lookup_by_fragment_hash_is_miss(self):
        """Looking up by fragment content_hash should be a miss;
        the manager keys by prefix_hash, not fragment hash."""
        mgr = KVCache()
        frag = make_fragment(content_hash="hit-hash")
        mgr.store_kv("prefix-a", [frag])
        result = mgr.lookup_kv("hit-hash")
        assert result == []
        assert mgr.get_miss_rate() == 1.0

    def test_multiple_fragments_under_same_prefix(self):
        mgr = KVCache()
        f1 = make_fragment(content_hash="f1")
        f2 = make_fragment(content_hash="f2")
        mgr.store_kv("prefix-a", [f1, f2])
        result = mgr.lookup_kv("prefix-a")
        assert len(result) == 2
        hashes = {f.identity.payload_hash for f in result}
        assert hashes == {"f1", "f2"}

    def test_store_overwrites_prefix_mapping(self):
        """Storing again under the same prefix should replace old fragments."""
        mgr = KVCache()
        f1 = make_fragment(content_hash="f1")
        f2 = make_fragment(content_hash="f2")
        mgr.store_kv("prefix-a", [f1])
        mgr.store_kv("prefix-a", [f2])
        result = mgr.lookup_kv("prefix-a")
        assert len(result) == 1
        assert result[0].identity.payload_hash == "f2"

    def test_remove_prefix(self):
        mgr = KVCache()
        frag = make_fragment(content_hash="f1")
        mgr.store_kv("prefix-a", [frag])
        assert mgr.remove_prefix("prefix-a") is True
        assert mgr.lookup_kv("prefix-a") == []
        assert mgr.remove_prefix("prefix-a") is False

    def test_metrics_snapshot(self):
        mgr = KVCache()
        mgr.lookup_kv("x")
        mgr.lookup_kv("y")
        metrics = mgr.get_metrics()
        assert isinstance(metrics, CacheMetrics)
        assert metrics.misses == 2
        assert metrics.total_requests == 2

    def test_store_with_custom_node_id(self):
        mgr = KVCache()
        frag = make_fragment(content_hash="custom")
        mgr.store_kv("p", [frag], node_id="node-7")
        entry = mgr.index_system.exact_lookup("custom")
        assert entry is not None
        assert "node-7" in entry.locations

    def test_hit_rate_after_mixed_access(self):
        mgr = KVCache()
        frag = make_fragment(content_hash="known")
        mgr.store_kv("p", [frag])
        mgr.lookup_kv("p")
        mgr.lookup_kv("unknown")
        assert mgr.get_hit_rate() == 0.5
        assert mgr.get_miss_rate() == 0.5

    def test_uses_provided_index_system(self):
        idx = Index()
        mgr = KVCache(index_system=idx)
        frag = make_fragment(content_hash="shared")
        mgr.store_kv("p", [frag])
        assert idx.exact_lookup("shared") is not None

    def test_lookup_returns_empty_list_not_none_on_miss(self):
        """lookup_kv must always return a list, never None."""
        mgr = KVCache()
        result = mgr.lookup_kv("nonexistent")
        assert result == []
        assert isinstance(result, list)

    def test_lru_eviction_on_overflow(self):
        mgr = KVCache(max_prefixes=2)
        mgr.store_kv("p1", [make_fragment(content_hash="a")])
        mgr.store_kv("p2", [make_fragment(content_hash="b")])
        mgr.store_kv("p3", [make_fragment(content_hash="c")])
        assert len(mgr.prefix_to_fragments) == 2
        assert "p1" not in mgr.prefix_to_fragments

    def test_lru_keeps_recently_accessed_prefix(self):
        mgr = KVCache(max_prefixes=2)
        mgr.store_kv("p1", [make_fragment(content_hash="a")])
        mgr.store_kv("p2", [make_fragment(content_hash="b")])
        mgr.lookup_kv("p1")  # make p1 recently used
        mgr.store_kv("p3", [make_fragment(content_hash="c")])
        assert "p1" in mgr.prefix_to_fragments
        assert "p2" not in mgr.prefix_to_fragments
