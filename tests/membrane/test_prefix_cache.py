"""Tests for the prefix cache + KV handle (Phase 7)."""

from __future__ import annotations

import threading

import pytest

from membrane.prefix_cache import KVHandle, PrefixCache, PrefixMatch


class TestKVHandle:
    def test_for_prefix_is_stable(self):
        a = KVHandle.for_prefix("m", (1, 2, 3))
        b = KVHandle.for_prefix("m", (1, 2, 3))
        assert a == b
        assert a.handle == b.handle
        assert a.token_len == 3
        assert a.model_id == "m"

    def test_for_prefix_changes_with_model(self):
        a = KVHandle.for_prefix("m1", (1, 2, 3))
        b = KVHandle.for_prefix("m2", (1, 2, 3))
        assert a.handle != b.handle

    def test_for_prefix_changes_with_tokens(self):
        a = KVHandle.for_prefix("m", (1, 2, 3))
        b = KVHandle.for_prefix("m", (1, 2, 4))
        assert a.handle != b.handle

    def test_for_tokens_accepts_list(self):
        a = KVHandle.for_tokens("m", [1, 2, 3])
        b = KVHandle.for_tokens("m", (1, 2, 3))
        assert a == b

    def test_handle_is_frozen(self):
        handle = KVHandle.for_prefix("m", (1,))
        with pytest.raises((AttributeError, TypeError)):
            handle.model_id = "other"  # type: ignore[misc]


class TestPrefixCache:
    def test_insert_and_lookup_full_hit(self):
        cache = PrefixCache()
        cache.insert("m", (1, 2, 3), layer_range=(0, 1))
        match = cache.lookup("m", (1, 2, 3))
        assert match.is_full
        assert match.token_len == 3
        assert match.handle is not None

    def test_lookup_longest_prefix(self):
        cache = PrefixCache()
        cache.insert("m", (1, 2), layer_range=(0, 0))
        cache.insert("m", (1, 2, 3), layer_range=(0, 1))
        match = cache.lookup("m", (1, 2, 3, 4))
        assert match.token_len == 3
        assert match.is_full is False

    def test_lookup_miss_on_different_model(self):
        cache = PrefixCache()
        cache.insert("m1", (1, 2, 3), layer_range=(0, 1))
        match = cache.lookup("m2", (1, 2, 3))
        assert match.handle is None
        assert match.token_len == 0

    def test_lookup_miss_on_different_prefix(self):
        cache = PrefixCache()
        cache.insert("m", (1, 2, 3), layer_range=(0, 1))
        match = cache.lookup("m", (4, 5, 6))
        assert match.handle is None

    def test_lookup_empty_tokens_is_miss(self):
        cache = PrefixCache()
        cache.insert("m", (1, 2, 3), layer_range=(0, 1))
        match = cache.lookup("m", ())
        assert match.handle is None

    def test_evict_removes_entry(self):
        cache = PrefixCache()
        handle = cache.insert("m", (1, 2, 3), layer_range=(0, 1))
        assert cache.evict(handle) is True
        match = cache.lookup("m", (1, 2, 3))
        assert match.handle is None

    def test_evict_missing_returns_false(self):
        cache = PrefixCache()
        handle = KVHandle.for_prefix("m", (1,))
        assert cache.evict(handle) is False

    def test_clear_drops_all_entries(self):
        cache = PrefixCache()
        cache.insert("m", (1,), layer_range=(0, 0))
        cache.insert("m", (1, 2), layer_range=(0, 0))
        cache.clear()
        assert cache.size() == 0

    def test_capacity_evicts_oldest(self):
        cache = PrefixCache(capacity=2)
        cache.insert("m", (1,), layer_range=(0, 0))
        cache.insert("m", (1, 2), layer_range=(0, 0))
        cache.insert("m", (1, 2, 3), layer_range=(0, 0))
        assert cache.size() == 2

    def test_zero_capacity_disables_cache(self):
        cache = PrefixCache(capacity=0)
        cache.insert("m", (1,), layer_range=(0, 0))
        assert cache.size() == 0

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            PrefixCache(capacity=-1)

    def test_stats_track_hits_and_misses(self):
        cache = PrefixCache()
        cache.insert("m", (1, 2, 3), layer_range=(0, 1))
        cache.lookup("m", (1, 2, 3))
        cache.lookup("m", (4, 5))
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_lookup_promotes_entry_to_most_recent(self):
        cache = PrefixCache(capacity=2)
        cache.insert("m", (1,), layer_range=(0, 0))
        cache.insert("m", (2,), layer_range=(0, 0))
        cache.lookup("m", (1,))
        cache.insert("m", (3,), layer_range=(0, 0))
        # (1,) was promoted; (2,) was the oldest and got evicted.
        assert cache.lookup("m", (1,)).handle is not None
        assert cache.lookup("m", (2,)).handle is None

    def test_concurrent_inserts_are_thread_safe(self):
        cache = PrefixCache(capacity=1000)
        threads = []

        def worker(offset: int) -> None:
            for i in range(50):
                cache.insert("m", (offset, i), layer_range=(0, 0))

        for offset in range(8):
            t = threading.Thread(target=worker, args=(offset,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert cache.size() == 8 * 50

    def test_reinsert_updates_layer_range(self):
        cache = PrefixCache()
        cache.insert("m", (1, 2, 3), layer_range=(0, 0))
        cache.insert("m", (1, 2, 3), layer_range=(0, 5))
        match = cache.lookup("m", (1, 2, 3))
        assert match.handle is not None
        assert match.handle.token_len == 3


class TestPrefixMatch:
    def test_miss_factory(self):
        match = PrefixMatch.miss()
        assert match.handle is None
        assert match.token_len == 0
        assert match.is_full is False
