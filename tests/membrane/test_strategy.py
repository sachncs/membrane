"""Tests for FailureDetector and Migrator strategies."""

import pytest

from membrane.network.strategy import (
    EagerMigrator,
    Migrator,
    QuorumDetector,
    RateLimitedMigrator,
    ThresholdDetector,
)


def test_threshold_detector_removes_at_threshold():
    """ThresholdDetector removes when missed >= threshold."""
    d = ThresholdDetector(failure_remove_threshold=4)
    assert not d.should_remove("p1", peer_missed=3, suspect_votes=0, healthy_peer_count=3)
    assert d.should_remove("p1", peer_missed=4, suspect_votes=0, healthy_peer_count=3)
    assert d.should_remove("p1", peer_missed=10, suspect_votes=0, healthy_peer_count=3)


def test_quorum_detector_requires_majority():
    """QuorumDetector requires majority + threshold."""
    d = QuorumDetector(failure_remove_threshold=4, suspect_threshold=1)
    # 3 healthy peers: quorum is 2.
    assert not d.should_remove("p1", peer_missed=4, suspect_votes=1, healthy_peer_count=3)
    assert d.should_remove("p1", peer_missed=4, suspect_votes=2, healthy_peer_count=3)
    # Below threshold of misses does not remove.
    assert not d.should_remove("p1", peer_missed=3, suspect_votes=3, healthy_peer_count=3)


def test_quorum_detector_falls_back_to_threshold_in_single_node():
    """Single-node clusters fall back to threshold semantics."""
    d = QuorumDetector()
    assert not d.should_remove("p1", peer_missed=3, suspect_votes=0, healthy_peer_count=1)
    assert d.should_remove("p1", peer_missed=4, suspect_votes=0, healthy_peer_count=1)


def test_eager_migrator_has_no_rate_limit():
    """EagerMigrator reports infinity migrations/second."""
    m = EagerMigrator()
    assert m.migrations_per_second() == float("inf")
    assert m.delay() == 0.0


def test_rate_limited_migrator_respects_cap():
    """RateLimitedMigrator delays between migrations."""
    m = RateLimitedMigrator(max_per_second=50.0)
    assert m.migrations_per_second() == 50.0
    assert 0.019 < m.delay() < 0.021  # ~1/50 = 0.02s


def test_rate_limited_migrator_zero_or_negative_treated_as_unlimited():
    """RateLimitedMigrator with invalid rate treats as unlimited."""
    m = RateLimitedMigrator(max_per_second=0.0)
    assert m.delay() == 0.0
