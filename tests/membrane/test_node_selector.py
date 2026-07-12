"""Tests for node_selector module."""

import pytest

from membrane.node_selector import NodeSelector, NodeSelectorConfig
from membrane.node_telemetry import NodeTelemetry


class TestNodeSelector:
    """Test suite for NodeSelector."""

    def test_select_empty_candidates(self):
        sel = NodeSelector()
        assert sel.select([], {}) == ""

    def test_select_missing_telemetry_skipped(self):
        sel = NodeSelector()
        assert sel.select(["n1"], {}) == ""

    def test_select_prefers_low_latency(self):
        sel = NodeSelector()
        telem = {
            "fast": NodeTelemetry("fast", 10.0, 0.0, 0.0, 0.0),
            "slow": NodeTelemetry("slow", 5000.0, 0.0, 0.0, 0.0),
        }
        best = sel.select(["fast", "slow"], telem)
        assert best == "fast"

    def test_select_prefers_low_load(self):
        sel = NodeSelector()
        telem = {
            "loaded": NodeTelemetry("loaded", 10.0, 0.0, 0.9, 0.9),
            "idle": NodeTelemetry("idle", 10.0, 0.0, 0.1, 0.1),
        }
        best = sel.select(["loaded", "idle"], telem)
        assert best == "idle"

    def test_health_filter_excludes_sick_nodes(self):
        cfg = NodeSelectorConfig(health_threshold=0.5)
        sel = NodeSelector(config=cfg)
        telem = {
            "sick": NodeTelemetry("sick", 10.0, 0.0, 0.9, 0.9),
            "healthy": NodeTelemetry("healthy", 10.0, 0.0, 0.1, 0.1),
        }
        best = sel.select(["sick", "healthy"], telem)
        assert best == "healthy"

    def test_all_unhealthy_returns_empty(self):
        cfg = NodeSelectorConfig(health_threshold=0.1)
        sel = NodeSelector(config=cfg)
        telem = {
            "n1": NodeTelemetry("n1", 10.0, 0.0, 0.9, 0.9),
        }
        assert sel.select(["n1"], telem) == ""

    def test_select_top_n(self):
        sel = NodeSelector()
        telem = {
            "a": NodeTelemetry("a", 10.0, 0.0, 0.0, 0.0),
            "b": NodeTelemetry("b", 100.0, 0.0, 0.0, 0.0),
            "c": NodeTelemetry("c", 1000.0, 0.0, 0.0, 0.0),
        }
        top = sel.select_top_n(["a", "b", "c"], telem, n=2)
        assert top == ["a", "b"]

    def test_weighted_config_changes_choice(self):
        cfg = NodeSelectorConfig(
            weight_latency=0.0,
            weight_gpu=1.0,
            weight_memory=0.0,
            weight_bandwidth=0.0,
            health_threshold=1.0,
        )
        sel = NodeSelector(config=cfg)
        telem = {
            "fast_but_overloaded": NodeTelemetry("fast_but_overloaded", 10.0, 0.0, 0.9, 0.0),
            "slow_but_idle": NodeTelemetry("slow_but_idle", 5000.0, 0.0, 0.1, 0.0),
        }
        best = sel.select(["fast_but_overloaded", "slow_but_idle"], telem)
        assert best == "slow_but_idle"

    def test_score_computation(self):
        sel = NodeSelector()
        telem = {
            "n1": NodeTelemetry("n1", 2500.0, 0.5, 0.5, 0.5),
        }
        score = sel.score("n1", telem)
        expected = 0.5 + 0.5 + 0.5 + 0.5  # all normalized to 0.5
        assert score == pytest.approx(expected)

    def test_score_missing_telemetry_inf(self):
        sel = NodeSelector()
        assert sel.score("n1", {}) == float("inf")

    def test_filter_healthy(self):
        sel = NodeSelector()
        telem = {
            "ok": NodeTelemetry("ok", 10.0, 0.0, 0.5, 0.5),
            "bad": NodeTelemetry("bad", 10.0, 0.0, 0.99, 0.99),
        }
        healthy = sel.filter_healthy(["ok", "bad"], telem)
        assert healthy == ["ok"]
