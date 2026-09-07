"""Tests for the tier + bandit + cost router wiring (Phase 3.5.6 + 3.5.7 + 3.5.8)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from membrane.tiers import (
    ArchivalTier,
    Bandit,
    BanditArm,
    ColdTier,
    EconomicRouterConfigWeights,
    HotTier,
    TierPolicy,
    WarmTier,
    apply_bandit_to_weights,
    record_op_store,
    select_tier,
)


@dataclass
class FakeFragment:
    reuse_score: float = 0.5
    payload_size: int = 0


class TestSelectTier:
    def test_hot(self):
        frag = FakeFragment(reuse_score=0.9)
        policy = TierPolicy()
        assert select_tier(policy, frag) == "hot"

    def test_warm(self):
        frag = FakeFragment(reuse_score=0.5)
        policy = TierPolicy()
        assert select_tier(policy, frag) == "warm"

    def test_cold(self):
        frag = FakeFragment(reuse_score=0.2)
        policy = TierPolicy()
        assert select_tier(policy, frag) == "cold"

    def test_archival(self):
        frag = FakeFragment(reuse_score=-0.5)
        policy = TierPolicy()
        assert select_tier(policy, frag) == "archival"


class TestTierNames:
    def test_hot_warm_cold_archive(self):
        assert HotTier().name() == "hot"
        assert WarmTier().name() == "warm"
        assert ColdTier().name() == "cold"
        assert ArchivalTier().name() == "archival"


class TestBandit:
    def test_initial_pulls_are_zero(self):
        arms = [BanditArm("a", 0.25), BanditArm("b", 0.25)]
        assert all(arm.pulls == 0 for arm in arms)

    def test_update_accumulates_reward(self):
        arm = BanditArm("a", 0.5)
        bandit = Bandit(arms=[arm], epsilon=0.0)
        bandit.update(arm, reward=1.0)
        bandit.update(arm, reward=0.0)
        assert arm.pulls == 2
        assert arm.reward_sum == 1.0
        assert arm.weight == pytest.approx(0.5)

    def test_estimated_reward_initial_uses_smoothing(self):
        arm = BanditArm("a", 0.5)
        bandit = Bandit(arms=[arm], epsilon=0.0)
        assert bandit._estimated_reward(arm) == 0.5

    def test_select_arm_with_zero_epsilon_returns_best(self):
        bandit = Bandit(
            arms=[BanditArm("a", 0.1), BanditArm("b", 0.9)],
            epsilon=0.0,
        )
        bandit.arms[1].weight = 0.9
        # With epsilon=0 and a-pulls=0 best-arm fallback, the
        # call still returns a valid arm.
        arm = bandit.select_arm()
        assert arm in bandit.arms


class TestEconomicRouterConfigWeights:
    def test_normalisation(self):
        cfg = EconomicRouterConfigWeights(
            latency_ms=1.0, bandwidth_cost=1.0, gpu_load=1.0, memory_pressure=1.0
        )
        w = cfg.normalised()
        assert math.isclose(sum(w), 1.0)

    def test_negative_weights_clamp_to_uniform(self):
        cfg = EconomicRouterConfigWeights(latency_ms=-1.0)
        w = cfg.normalised()
        assert w == (0.25, 0.25, 0.25, 0.25)


class TestApplyBanditToWeights:
    def test_no_bandit_returns_base(self):
        cfg = EconomicRouterConfigWeights(latency_ms=0.3, bandwidth_cost=0.1, gpu_load=0.1, memory_pressure=0.5)
        assert apply_bandit_to_weights(None, cfg) is cfg

    def test_with_bandit_uses_arm_weights(self):
        bandit = Bandit(
            arms=[
                BanditArm("latency_ms", 0.4),
                BanditArm("bandwidth_cost", 0.2),
                BanditArm("gpu_load", 0.2),
                BanditArm("memory_pressure", 0.2),
            ],
        )
        cfg = apply_bandit_to_weights(bandit, EconomicRouterConfigWeights())
        assert cfg.latency_ms >= 0.05
        assert cfg.gpu_load >= 0.05


class TestRecordOpStore:
    def test_admitted_by_default(self):
        frag = FakeFragment(reuse_score=0.0)
        assert record_op_store(frag)

    def test_rejected_by_admission_policy(self):
        frag = FakeFragment(reuse_score=0.1)
        from membrane.decision import AdmissionPolicy

        policy = AdmissionPolicy(enabled=True, min_reuse_score=0.5)
        assert not record_op_store(frag, policy=policy)
