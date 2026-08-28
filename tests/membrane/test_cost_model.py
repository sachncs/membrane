"""Tests for cost_model module."""

import pytest

from membrane.cost import CostModel


class TestCostModel:
    """Test suite for CostModel."""

    def test_reuse_is_cheaper_when_retrieval_low(self):
        model = CostModel(bandwidth_gbps=1000.0)
        assert model.reuse_is_cheaper(prefix_length=2048, kv_size=10.0)

    def test_reuse_is_not_cheaper_when_compute_low(self):
        model = CostModel(bandwidth_gbps=0.01)
        assert not model.reuse_is_cheaper(prefix_length=10, kv_size=1000.0)

    def test_reuse_with_explicit_latency_override(self):
        model = CostModel()
        assert model.reuse_is_cheaper(prefix_length=2048, kv_size=1000.0, retrieval_latency_seconds=0.01)

    def test_retrieval_cost_zero_bandwidth_returns_inf(self):
        model = CostModel(bandwidth_gbps=0.0)
        assert model.find_cost(10.0) == float("inf")

    def test_retrieval_cost_negative_bandwidth_returns_inf(self):
        model = CostModel(bandwidth_gbps=-1.0)
        assert model.find_cost(10.0) == float("inf")

    def test_precompute_cost_monotonic(self):
        model = CostModel()
        c1 = model.prefill_cost(1024)
        c2 = model.prefill_cost(2048)
        assert c2 >= c1

    def test_reuse_is_not_cheaper_when_override_high(self):
        model = CostModel()
        assert not model.reuse_is_cheaper(prefix_length=10, kv_size=1.0, retrieval_latency_seconds=100.0)

    def test_scale_affects_compute_cost(self):
        model_fast = CostModel(compute_scale=0.5)
        model_slow = CostModel(compute_scale=2.0)
        assert model_fast.prefill_cost(1024) < model_slow.prefill_cost(1024)

    def test_retrieval_cost_increases_with_size(self):
        model = CostModel(bandwidth_gbps=10.0)
        small = model.find_cost(1.0)
        large = model.find_cost(100.0)
        assert large > small
