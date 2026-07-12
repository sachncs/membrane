"""Tests for fragment_store module."""

import time

import pytest

from membrane.fragment import Fragment
from membrane.fragment_store import FragmentStore
from membrane.structural_signature import StructuralSignature


def make_fragment(
    content_hash: str,
    size: int = 10,
    reuse_score: float = 0.5,
    ttl: float = 3600.0,
):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=size,
        ttl=ttl,
        reuse_score=reuse_score,
        version_id=1,
    )


class TestFragmentStore:
    """Test suite for FragmentStore."""

    def test_put_and_get(self):
        store = FragmentStore()
        frag = make_fragment("h1")
        assert store.put(frag)
        assert store.get("h1") == frag

    def test_get_missing_returns_none(self):
        store = FragmentStore()
        assert store.get("missing") is None
        assert store.metrics.miss_count == 1

    def test_get_updates_access_time(self):
        store = FragmentStore()
        store.put(make_fragment("h1"))
        t0 = store.access_times["h1"]
        time.sleep(0.01)
        store.get("h1")
        assert store.access_times["h1"] > t0

    def test_put_duplicate_refreshes(self):
        store = FragmentStore()
        store.put(make_fragment("h1"))
        assert store.put(make_fragment("h1"))
        assert store.metrics.stored_count == 1

    def test_put_rejects_too_large(self):
        store = FragmentStore(max_bytes=50)
        frag = make_fragment("big", size=100)
        assert not store.put(frag)

    def test_ttl_expiry_on_get(self):
        store = FragmentStore()
        store.put(make_fragment("old", ttl=0.01))
        time.sleep(0.02)
        assert store.get("old") is None
        assert store.metrics.expired_count == 1

    def test_remove(self):
        store = FragmentStore()
        store.put(make_fragment("h1"))
        removed = store.remove("h1")
        assert removed is not None
        assert removed.content_hash == "h1"
        assert store.get("h1") is None

    def test_evict_respects_max_count(self):
        store = FragmentStore(max_count=2)
        store.put(make_fragment("a"))
        store.put(make_fragment("b"))
        store.put(make_fragment("c"))
        assert store.metrics.stored_count == 2

    def test_evict_respects_max_bytes(self):
        store = FragmentStore(max_bytes=30)
        store.put(make_fragment("a", size=10))
        store.put(make_fragment("b", size=10))
        store.put(make_fragment("c", size=10))
        store.put(make_fragment("d", size=10))
        assert store.metrics.stored_bytes == 30

    def test_tier_classification(self):
        store = FragmentStore(hot_ttl=1.0, warm_ttl=5.0)
        store.put(make_fragment("h1"))
        assert store.tier("h1") == "hot"

    def test_tier_warm(self):
        store = FragmentStore(hot_ttl=0.01, warm_ttl=5.0)
        store.put(make_fragment("h1"))
        time.sleep(0.02)
        assert store.tier("h1") == "warm"

    def test_tier_cold(self):
        store = FragmentStore(hot_ttl=0.01, warm_ttl=0.02)
        store.put(make_fragment("h1"))
        time.sleep(0.03)
        assert store.tier("h1") == "cold"

    def test_tier_counts(self):
        store = FragmentStore(hot_ttl=1.0, warm_ttl=5.0)
        store.put(make_fragment("a"))
        store.put(make_fragment("b"))
        assert store.tier_counts() == {"hot": 2, "warm": 0, "cold": 0}

    def test_evict_one_prefers_expired(self):
        store = FragmentStore()
        store.put(make_fragment("expired", ttl=0.01))
        store.put(make_fragment("fresh", ttl=3600.0))
        time.sleep(0.02)
        evicted = store.evict_one()
        assert evicted is not None
        assert evicted.content_hash == "expired"

    def test_evict_one_prefers_cold_low_value(self):
        store = FragmentStore(hot_ttl=1.0, warm_ttl=5.0)
        store.put(make_fragment("cold_low", reuse_score=0.1))
        time.sleep(1.1)
        store.put(make_fragment("hot_high", reuse_score=0.9))
        evicted = store.evict_one()
        assert evicted is not None
        assert evicted.content_hash == "cold_low"

    def test_evict_to_target_frees_bytes(self):
        store = FragmentStore(max_bytes=50)
        store.put(make_fragment("a", size=20))
        store.put(make_fragment("b", size=20))
        freed_hashes = store.evict_to_target(15)
        assert len(freed_hashes) >= 1
        assert store.metrics.stored_bytes <= 50

    def test_hit_rate(self):
        store = FragmentStore()
        store.put(make_fragment("h1"))
        store.get("h1")
        store.get("missing")
        assert store.get_hit_rate() == 0.5
        assert store.get_miss_rate() == 0.5

    def test_values_and_keys(self):
        store = FragmentStore()
        store.put(make_fragment("a"))
        store.put(make_fragment("b"))
        assert len(store.values()) == 2
        assert store.keys() == {"a", "b"}

    def test_metrics_track_limits(self):
        store = FragmentStore(max_bytes=100, max_count=5)
        assert store.metrics.max_bytes == 100
        assert store.metrics.max_count == 5

    def test_store_get_hit_count(self):
        store = FragmentStore()
        store.put(make_fragment("h1"))
        store.get("h1")
        store.get("h1")
        assert store.metrics.hit_count == 2
