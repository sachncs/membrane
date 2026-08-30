from tests.conftest import make_fragment

"""Tests for promotion_policy module."""

import pytest

from membrane.policy import Promotion, PromotionConfig, PromotionResult


class TestPromotionPolicy:
    """Test suite for Promotion."""

    def test_low_reuse_score_no_promote(self):
        policy = Promotion()
        frag = make_fragment(reuse_score=0.1)
        decision = policy.evaluate(frag, {"us": 5}, [])
        assert not decision.should_promote
        assert decision.reason == "reuse_score below threshold"

    def test_low_demand_no_promote(self):
        policy = Promotion()
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 1}, [])
        assert not decision.should_promote
        assert decision.reason == "demand below threshold"

    def test_max_replicas_reached(self):
        policy = Promotion(config=PromotionConfig(max_replicas=2))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 5, "eu": 5}, ["r1", "r2"])
        assert not decision.should_promote
        assert decision.reason == "max replicas reached"

    def test_promote_to_top_region(self):
        policy = Promotion(config=PromotionConfig(max_replicas=2))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 10, "eu": 3}, [])
        assert decision.should_promote
        assert "us" in decision.target_replicas
        assert decision.reason == "high reuse and multi-region demand"

    def test_promote_respects_existing_replicas(self):
        policy = Promotion(config=PromotionConfig(max_replicas=3))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 10, "eu": 8, "ap": 5}, ["us"])
        assert decision.should_promote
        assert "us" not in decision.target_replicas
        assert len(decision.target_replicas) <= 2

    def test_no_suitable_regions_when_all_existing(self):
        policy = Promotion(config=PromotionConfig(max_replicas=2))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 5}, ["us"])
        assert not decision.should_promote
        assert decision.reason == "no suitable regions"

    def test_promote_multiple_regions(self):
        policy = Promotion(config=PromotionConfig(max_replicas=3))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 10, "eu": 8, "ap": 6, "sa": 4}, [])
        assert decision.should_promote
        assert len(decision.target_replicas) == 3
        assert decision.target_replicas == ["us", "eu", "ap"]
