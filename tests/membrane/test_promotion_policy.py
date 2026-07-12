"""Tests for promotion_policy module."""

import pytest

from membrane.fragment import Fragment
from membrane.promotion_policy import PromotionConfig, PromotionDecision, PromotionPolicy
from membrane.structural_signature import StructuralSignature


def make_fragment(reuse_score=0.8):
    return Fragment(
        content_hash="abc",
        embedding=(0.0,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=10,
        ttl=3600.0,
        reuse_score=reuse_score,
        version_id=1,
    )


class TestPromotionPolicy:
    """Test suite for PromotionPolicy."""

    def test_low_reuse_score_no_promote(self):
        policy = PromotionPolicy()
        frag = make_fragment(reuse_score=0.1)
        decision = policy.evaluate(frag, {"us": 5}, [])
        assert not decision.should_promote
        assert decision.reason == "reuse_score below threshold"

    def test_low_demand_no_promote(self):
        policy = PromotionPolicy()
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 1}, [])
        assert not decision.should_promote
        assert decision.reason == "demand below threshold"

    def test_max_replicas_reached(self):
        policy = PromotionPolicy(config=PromotionConfig(max_replicas=2))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 5, "eu": 5}, ["r1", "r2"])
        assert not decision.should_promote
        assert decision.reason == "max replicas reached"

    def test_promote_to_top_region(self):
        policy = PromotionPolicy(config=PromotionConfig(max_replicas=2))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 10, "eu": 3}, [])
        assert decision.should_promote
        assert "us" in decision.target_replicas
        assert decision.reason == "high reuse and multi-region demand"

    def test_promote_respects_existing_replicas(self):
        policy = PromotionPolicy(config=PromotionConfig(max_replicas=3))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 10, "eu": 8, "ap": 5}, ["us"])
        assert decision.should_promote
        assert "us" not in decision.target_replicas
        assert len(decision.target_replicas) <= 2

    def test_no_suitable_regions_when_all_existing(self):
        policy = PromotionPolicy(config=PromotionConfig(max_replicas=2))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 5}, ["us"])
        assert not decision.should_promote
        assert decision.reason == "no suitable regions"

    def test_promote_multiple_regions(self):
        policy = PromotionPolicy(config=PromotionConfig(max_replicas=3))
        frag = make_fragment(reuse_score=0.9)
        decision = policy.evaluate(frag, {"us": 10, "eu": 8, "ap": 6, "sa": 4}, [])
        assert decision.should_promote
        assert len(decision.target_replicas) == 3
        assert decision.target_replicas == ["us", "eu", "ap"]
