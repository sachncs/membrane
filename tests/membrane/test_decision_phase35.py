"""Tests for the v3.0 admission + TinyLFU + quota + EMA + prefetcher (Phase 3.5.1-3.5.5 + 3.5.9)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from membrane.decision import (
    AdmissionPolicy,
    CoaccessSessionPrefetcher,
    HitObserver,
    PredictHitProbability,
    TenantQuota,
    TinyLFU,
)


@dataclass
class FakeFragment:
    reuse_score: float = 0.0


class _FakeCoaccess:
    """CoaccessIndex stub that returns a fixed neighbor list."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def neighbors(self, key: str):
        return self.mapping.get(key, [])


class _FakeCache:
    """Cache stub with a get_or_load method that records touches."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    def get_or_load(self, key: str) -> None:
        self.touched.append(key)


class TestAdmissionPolicy:
    def test_disabled_admits_everything(self):
        policy = AdmissionPolicy(enabled=False)
        assert policy.should_admit(0.0)
        assert policy.should_admit(1.0)

    def test_enabled_threshold(self):
        policy = AdmissionPolicy(enabled=True, min_reuse_score=0.5)
        assert policy.should_admit(0.6) is True
        assert policy.should_admit(0.5) is True
        assert policy.should_admit(0.4) is False

    def test_zero_threshold_admits_every_positive(self):
        policy = AdmissionPolicy(enabled=True, min_reuse_score=0.0)
        assert policy.should_admit(0.0)
        assert policy.should_admit(1.0)


class TestTinyLFU:
    def test_construction(self):
        cache = TinyLFU(capacity=10, window_ratio=0.1)
        assert cache.size() == 0

    def test_construction_rejects_zero_capacity(self):
        with pytest.raises(ValueError):
            TinyLFU(capacity=0)

    def test_admit_and_touch(self):
        cache = TinyLFU(capacity=10, window_ratio=0.1)
        for i in range(5):
            cache.admit(f"key-{i}")
        assert cache.size() == 5
        cache.touch("key-0")
        # Touched keys move into the main segment; the cache
        # still holds all 5.
        assert cache.size() == 5

    def test_evict_drops_key(self):
        cache = TinyLFU(capacity=4, window_ratio=0.0)
        cache.admit("a")
        cache.admit("b")
        cache.evict("a")
        assert cache.size() == 1

    def test_freq_sketch_increments(self):
        cache = TinyLFU(capacity=4)
        cache.touch("alpha")
        cache.touch("alpha")
        cache.touch("alpha")
        # The sketch's hash bucket for "alpha" should have a
        # non-zero count.
        assert cache._estimate("alpha") >= 3


class TestTenantQuota:
    def test_unlimited_admits(self):
        quota = TenantQuota(tenant_id="acme")
        assert quota.admit(1000) is True

    def test_byte_cap_rejects(self):
        quota = TenantQuota(tenant_id="acme", max_bytes=100)
        assert quota.admit(60) is True
        assert quota.admit(60) is False

    def test_entry_cap_rejects(self):
        quota = TenantQuota(tenant_id="acme", max_entries=2)
        assert quota.admit(10) is True
        assert quota.admit(10) is True
        assert quota.admit(10) is False

    def test_release_refunds_bytes(self):
        quota = TenantQuota(tenant_id="acme", max_bytes=200)
        quota.admit(60)
        quota.admit(50)
        assert quota.used_bytes == 110
        quota.release(50)
        assert quota.used_bytes == 60


class TestHitObserver:
    def test_increments_reuse_score(self):
        observer = HitObserver(alpha=0.5, decay=0.0)
        frag = FakeFragment(reuse_score=0.0)
        observer.record_hit(frag)
        assert frag.reuse_score == pytest.approx(0.5)
        observer.record_hit(frag)
        assert frag.reuse_score == pytest.approx(0.75)

    def test_decay_reduces_observation(self):
        observer = HitObserver(alpha=0.5, decay=0.2)
        frag = FakeFragment(reuse_score=0.5)
        observer.record_hit(frag)
        # observed = 1 - 0.2 = 0.8; new = 0.5*0.8 + 0.5*0.5 = 0.65
        assert frag.reuse_score == pytest.approx(0.65)


class TestCoaccessSessionPrefetcher:
    def test_predict_with_empty_history(self):
        cache = _FakeCache()
        coa = _FakeCoaccess({})
        prefetcher = CoaccessSessionPrefetcher(cache=cache, coaccess=coa)
        prediction = prefetcher.predict_next("alpha")
        assert prediction.probability == 0.0

    def test_predict_after_history(self):
        cache = _FakeCache()
        coa = _FakeCoaccess({})
        prefetcher = CoaccessSessionPrefetcher(cache=cache, coaccess=coa)
        for _ in range(3):
            prefetcher.record_access("alpha")
        prediction = prefetcher.predict_next("alpha")
        assert prediction.probability == pytest.approx(1.0)

    def test_record_access_prefetches_neighbors(self):
        cache = _FakeCache()
        coa = _FakeCoaccess({"alpha": ["beta", "gamma"]})
        prefetcher = CoaccessSessionPrefetcher(cache=cache, coaccess=coa)
        prefetcher.record_access("alpha")
        assert "beta" in cache.touched
        assert "gamma" in cache.touched

    def test_seen_set_avoids_re_prefetch(self):
        cache = _FakeCache()
        coa = _FakeCoaccess({"alpha": ["beta"]})
        prefetcher = CoaccessSessionPrefetcher(cache=cache, coaccess=coa)
        prefetcher.record_access("alpha")
        prefetcher.record_access("alpha")
        prefetcher.record_access("alpha")
        # beta was touched only once.
        assert cache.touched.count("beta") == 1
