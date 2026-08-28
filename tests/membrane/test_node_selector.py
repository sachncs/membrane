"""Tests for node_selector module."""

import pytest

from membrane.selector import Selector, SelectorConfig
from membrane.telemetry import Telemetry


class TestNodeSelector:
    """Test suite for Selector."""

    def test_select_empty_candidates(self):
        sel = Selector()
        assert sel.select([], {}) == ""

    def test_select_missing_telemetry_skipped(self):
        sel = Selector()
        assert sel.select(["n1"], {}) == ""

    def test_select_prefers_low_latency(self):
        sel = Selector()
        telem = {
            "fast": Telemetry("fast", 10.0, 0.0, 0.0, 0.0),
            "slow": Telemetry("slow", 5000.0, 0.0, 0.0, 0.0),
        }
        best = sel.select(["fast", "slow"], telem)
        assert best == "fast"

    def test_select_prefers_low_load(self):
        sel = Selector()
        telem = {
            "loaded": Telemetry("loaded", 10.0, 0.0, 0.9, 0.9),
            "idle": Telemetry("idle", 10.0, 0.0, 0.1, 0.1),
        }
        best = sel.select(["loaded", "idle"], telem)
        assert best == "idle"

    def test_health_filter_excludes_sick_nodes(self):
        cfg = SelectorConfig(health_threshold=0.5)
        sel = Selector(config=cfg)
        telem = {
            "sick": Telemetry("sick", 10.0, 0.0, 0.9, 0.9),
            "healthy": Telemetry("healthy", 10.0, 0.0, 0.1, 0.1),
        }
        best = sel.select(["sick", "healthy"], telem)
        assert best == "healthy"

    def test_all_unhealthy_returns_empty(self):
        cfg = SelectorConfig(health_threshold=0.1)
        sel = Selector(config=cfg)
        telem = {
            "n1": Telemetry("n1", 10.0, 0.0, 0.9, 0.9),
        }
        assert sel.select(["n1"], telem) == ""

    def test_select_top_n(self):
        sel = Selector()
        telem = {
            "a": Telemetry("a", 10.0, 0.0, 0.0, 0.0),
            "b": Telemetry("b", 100.0, 0.0, 0.0, 0.0),
            "c": Telemetry("c", 1000.0, 0.0, 0.0, 0.0),
        }
        top = sel.select_top_n(["a", "b", "c"], telem, n=2)
        assert top == ["a", "b"]

    def test_weighted_config_changes_choice(self):
        cfg = SelectorConfig(
            weight_latency=0.0,
            weight_gpu=1.0,
            weight_memory=0.0,
            weight_bandwidth=0.0,
            health_threshold=1.0,
        )
        sel = Selector(config=cfg)
        telem = {
            "fast_but_overloaded": Telemetry("fast_but_overloaded", 10.0, 0.0, 0.9, 0.0),
            "slow_but_idle": Telemetry("slow_but_idle", 5000.0, 0.0, 0.1, 0.0),
        }
        best = sel.select(["fast_but_overloaded", "slow_but_idle"], telem)
        assert best == "slow_but_idle"

    def test_score_computation(self):
        sel = Selector()
        telem = {
            "n1": Telemetry("n1", 2500.0, 0.5, 0.5, 0.5),
        }
        score = sel.score("n1", telem)
        expected = 0.5 + 0.5 + 0.5 + 0.5  # all normalized to 0.5
        assert score == pytest.approx(expected)

    def test_score_missing_telemetry_inf(self):
        sel = Selector()
        assert sel.score("n1", {}) == float("inf")

    def test_filter_healthy(self):
        sel = Selector()
        telem = {
            "ok": Telemetry("ok", 10.0, 0.0, 0.5, 0.5),
            "bad": Telemetry("bad", 10.0, 0.0, 0.99, 0.99),
        }
        healthy = sel.filter_healthy(["ok", "bad"], telem)
        assert healthy == ["ok"]
