"""Tests for cache_metrics module."""

import pytest

from membrane.cache_metrics import CacheMetrics


class TestCacheMetrics:
    """Test suite for CacheMetrics."""

    def test_initial_state(self):
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total_requests == 0
        assert metrics.total_kv_size_bytes == 0
        assert metrics.peak_memory_bytes == 0

    def test_record_hit(self):
        m = CacheMetrics()
        updated = m.record_hit()
        assert updated.hits == 1
        assert updated.total_requests == 1
        assert updated.misses == 0

    def test_record_miss(self):
        m = CacheMetrics()
        updated = m.record_miss(kv_size_bytes=1024)
        assert updated.misses == 1
        assert updated.total_requests == 1
        assert updated.total_kv_size_bytes == 1024
        assert updated.peak_memory_bytes == 1024

    def test_hit_rate_empty(self):
        assert CacheMetrics().hit_rate() == 0.0

    def test_hit_rate_half(self):
        m = CacheMetrics(hits=3, misses=3, total_requests=6)
        assert m.hit_rate() == 0.5

    def test_miss_rate(self):
        m = CacheMetrics(hits=1, misses=3, total_requests=4)
        assert m.miss_rate() == 0.75

    def test_size_growth_rate_empty(self):
        assert CacheMetrics().size_growth_rate() == 0.0

    def test_size_growth_rate_computed(self):
        m = CacheMetrics(misses=2, total_kv_size_bytes=2048)
        assert m.size_growth_rate() == 1024.0

    def test_immutability_after_multiple_updates(self):
        m = CacheMetrics()
        m1 = m.record_hit()
        m2 = m1.record_miss(kv_size_bytes=100)
        m3 = m2.record_hit()
        assert m.hits == 0
        assert m1.hits == 1
        assert m2.misses == 1
        assert m3.hits == 2
        assert m3.total_requests == 3

    def test_peak_memory_tracks_max(self):
        m = CacheMetrics()
        m1 = m.record_miss(kv_size_bytes=100)
        m2 = m1.record_miss(kv_size_bytes=50)
        assert m2.peak_memory_bytes == 150
        m3 = m2.record_miss(kv_size_bytes=200)
        assert m3.peak_memory_bytes == 350
