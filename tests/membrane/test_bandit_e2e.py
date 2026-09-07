"""End-to-end test for the Bandit online-learning loop (Phase 3.5.8 follow-up)."""

from __future__ import annotations

import pytest

from membrane.tiers import Bandit, BanditArm, apply_bandit_to_weights


class TestBanditE2E:
    def test_bandit_converges_toward_high_reward_arm(self):
        """Many updates on one arm should bias select_arm toward it."""
        arms = [
            BanditArm(name="a", weight=0.5),
            BanditArm(name="b", weight=0.5),
            BanditArm(name="c", weight=0.5),
        ]
        bandit = Bandit(arms=arms, epsilon=0.0)  # pure exploitation
        # Give arm "a" 50 winning updates; "b" and "c" stay at zero.
        for _ in range(50):
            bandit.update(arms[0], reward=1.0)
        # The estimated reward for arm a is 1.0; for others 0.
        assert bandit._estimated_reward(arms[0]) == 1.0
        # Without exploration, select_arm should return arm a.
        assert bandit.select_arm() is arms[0]

    def test_epsilon_exploration_visits_other_arms(self):
        arms = [
            BanditArm(name="a", weight=0.5),
            BanditArm(name="b", weight=0.5),
            BanditArm(name="c", weight=0.5),
        ]
        bandit = Bandit(arms=arms, epsilon=1.0)  # pure exploration
        # With pure exploration every select_arm is uniform random;
        # across 100 draws we should see all three arms at least once.
        for _ in range(100):
            assert bandit.select_arm() in arms
        seen = {bandit.select_arm().name for _ in range(300)}
        # With epsilon=1.0 over 300 draws, all three are likely visited.
        assert seen == {"a", "b", "c"}

    def test_apply_bandit_to_weights_returns_normalized(self):
        from membrane.tiers import EconomicRouterConfigWeights

        arms = [
            BanditArm(name="latency_ms", weight=0.4),
            BanditArm(name="bandwidth_cost", weight=0.3),
            BanditArm(name="gpu_load", weight=0.2),
            BanditArm(name="memory_pressure", weight=0.1),
        ]
        cfg = apply_bandit_to_weights(Bandit(arms=arms), EconomicRouterConfigWeights())
        # Sum should be normalized.
        total = (
            cfg.latency_ms + cfg.bandwidth_cost + cfg.gpu_load + cfg.memory_pressure
        )
        assert total == pytest.approx(1.0)

    def test_pull_count_tracks_pulls(self):
        arm = BanditArm(name="a", weight=0.5)
        bandit = Bandit(arms=[arm], epsilon=0.0)
        assert arm.pulls == 0
        bandit.select_arm()  # selection doesn't increment pulls
        assert arm.pulls == 0
        bandit.update(arm, reward=1.0)
        assert arm.pulls == 1
        bandit.update(arm, reward=1.0)
        assert arm.pulls == 2

    def test_no_bandit_returns_base_unchanged(self):
        from membrane.tiers import EconomicRouterConfigWeights

        cfg = EconomicRouterConfigWeights(
            latency_ms=0.4, bandwidth_cost=0.3, gpu_load=0.2, memory_pressure=0.1
        )
        result = apply_bandit_to_weights(None, cfg)
        assert result is cfg

    def test_bandit_short_arms_returns_base(self):
        from membrane.tiers import EconomicRouterConfigWeights

        bandit = Bandit(arms=[BanditArm("a", 0.5)])  # only 1 arm
        cfg = EconomicRouterConfigWeights()
        result = apply_bandit_to_weights(bandit, cfg)
        assert result is cfg
