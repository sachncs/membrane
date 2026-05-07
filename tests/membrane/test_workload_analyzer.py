"""Tests for workload_analyzer module."""

import pytest

from membrane.workload_analyzer import WorkloadAnalyzer


class TestWorkloadAnalyzer:
    """Test suite for WorkloadAnalyzer."""

    def test_analyze_patterns_empty(self):
        wa = WorkloadAnalyzer()
        assert wa.analyze_patterns([]) == {}

    def test_analyze_patterns_frequencies(self):
        wa = WorkloadAnalyzer()
        log = ["a", "b", "a"]
        freqs = wa.analyze_patterns(log)
        assert freqs["a"] == pytest.approx(2 / 3)
        assert freqs["b"] == pytest.approx(1 / 3)

    def test_top_patterns_sorted(self):
        wa = WorkloadAnalyzer()
        log = ["a"] * 5 + ["b"] * 3 + ["c"] * 1
        top = wa.top_patterns(log, k=2)
        assert top[0][0] == "a"
        assert top[1][0] == "b"
        assert len(top) == 2

    def test_reuse_ratio_all_unique(self):
        wa = WorkloadAnalyzer()
        assert wa.reuse_ratio(["a", "b", "c"]) == 0.0

    def test_reuse_ratio_all_same(self):
        wa = WorkloadAnalyzer()
        assert wa.reuse_ratio(["a", "a", "a"]) == pytest.approx(2 / 3)

    def test_reuse_ratio_empty(self):
        wa = WorkloadAnalyzer()
        assert wa.reuse_ratio([]) == 0.0

    def test_top_patterns_k_larger_than_unique(self):
        wa = WorkloadAnalyzer()
        log = ["a", "b"]
        top = wa.top_patterns(log, k=10)
        assert len(top) == 2
