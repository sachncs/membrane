"""Tests for economic_router module."""

import pytest

from membrane.economic_router import EconomicRouter, EconomicRouterConfig
from membrane.fragment import Fragment
from membrane.node_telemetry import NodeTelemetry
from membrane.structural_signature import StructuralSignature


def make_fragment(reuse_score=0.5):
    return Fragment(
        content_hash="abc",
        embedding=(0.0,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=10,
        ttl=3600.0,
        reuse_score=reuse_score,
        version_id=1,
    )


class TestEconomicRouter:
    """Test suite for EconomicRouter."""

    def test_empty_candidates_returns_empty(self):
        router = EconomicRouter()
        frag = make_fragment()
        assert router.route(frag, [], {}, []) == ""

    def test_selects_lowest_cost_node(self):
        router = EconomicRouter()
        frag = make_fragment(reuse_score=1.0)
        telemetry = {
            "n1": NodeTelemetry(
                "n1",
                latency_ms=1000.0,
                bandwidth_cost=0.5,
                gpu_load=0.8,
                memory_pressure=0.8,
            ),
            "n2": NodeTelemetry(
                "n2",
                latency_ms=10.0,
                bandwidth_cost=0.1,
                gpu_load=0.1,
                memory_pressure=0.1,
            ),
        }
        best = router.route(frag, ["n1", "n2"], telemetry, [])
        assert best == "n2"

    def test_missing_telemetry_negative_inf(self):
        router = EconomicRouter()
        frag = make_fragment(reuse_score=1.0)
        telemetry = {"n1": NodeTelemetry("n1", 0.0, 0.0, 0.0, 0.0)}
        best = router.route(frag, ["n1", "n2"], telemetry, [])
        assert best == "n1"

    def test_value_density_affects_choice(self):
        router = EconomicRouter()
        high_value = make_fragment(reuse_score=0.9)
        telemetry = {
            "n1": NodeTelemetry("n1", 0.0, 0.0, 0.0, 0.0),
        }
        best = router.route(high_value, ["n1"], telemetry, ["h", "h"])
        assert best == "n1"

    def test_latency_normalization_prevents_domination(self):
        """Without normalization, a 5000 ms latency would dominate the score.
        With normalization capped at 1.0, it should not overwhelm other dims."""
        router = EconomicRouter(config=EconomicRouterConfig(max_latency_ms=1000.0))
        frag = make_fragment(reuse_score=0.5)
        # n1: latency 5000 -> normalized to 1.0, but everything else is 0
        # n2: latency 100, bandwidth 0.5, gpu 0.5, memory 0.5
        telemetry = {
            "n1": NodeTelemetry("n1", 5000.0, 0.0, 0.0, 0.0),
            "n2": NodeTelemetry("n2", 100.0, 0.5, 0.5, 0.5),
        }
        # n1 cost = 1.0 * 1.0 = 1.0
        # n2 cost = 0.1 * 1.0 + 0.5 + 0.5 + 0.5 = 1.6
        # So n1 should win despite the huge raw latency
        best = router.route(frag, ["n1", "n2"], telemetry, [])
        assert best == "n1"

    def test_weighted_config_changes_outcome(self):
        """By zeroing latency weight, a high-latency low-load node can win."""
        frag = make_fragment(reuse_score=0.5)
        telemetry = {
            "fast_but_overloaded": NodeTelemetry("fast_but_overloaded", 10.0, 0.0, 0.9, 0.9),
            "slow_but_idle": NodeTelemetry("slow_but_idle", 5000.0, 0.0, 0.1, 0.1),
        }
        # With default weights, fast_but_overloaded should lose because of
        # high gpu/memory cost.
        router_default = EconomicRouter()
        best = router_default.route(frag, ["fast_but_overloaded", "slow_but_idle"], telemetry, [])
        assert best == "slow_but_idle"

        # With zero gpu/memory weight, fast node should win
        router_latency_only = EconomicRouter(
            config=EconomicRouterConfig(
                weight_latency=1.0,
                weight_gpu=0.0,
                weight_memory=0.0,
                weight_bandwidth=0.0,
            )
        )
        best2 = router_latency_only.route(frag, ["fast_but_overloaded", "slow_but_idle"], telemetry, [])
        assert best2 == "fast_but_overloaded"

    def test_clamped_values_stay_in_range(self):
        """Telemetry values outside [0, 1] should be clamped."""
        router = EconomicRouter()
        frag = make_fragment()
        telemetry = {
            "n1": NodeTelemetry("n1", 100_000.0, 5.0, 2.0, -1.0),
        }
        # Should not crash; negative values should be handled gracefully
        best = router.route(frag, ["n1"], telemetry, [])
        assert best == "n1"
