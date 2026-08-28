"""Tests for offload_decision_engine module."""

import pytest

from membrane.cost import CostModel
from membrane.node import Node
from membrane.offload import (
    Offload,
    OffloadConfig,
    OffloadResult,
)


class TestOffloadDecisionEngine:
    """Test suite for Offload."""

    def test_short_prompt_low_load_local(self):
        engine = Offload()
        local = Node("local")
        decision = engine.decide(list(range(100)), local, [])
        assert decision.local_compute
        assert decision.target_node_id == "local"
        assert "short" in decision.reason

    def test_long_prompt_offloads(self):
        engine = Offload()
        local = Node("local")
        remote = Node("remote")
        decision = engine.decide(list(range(2048)), local, [remote])
        assert not decision.local_compute
        assert decision.target_node_id == "remote"

    def test_high_local_load_offloads(self):
        config = OffloadConfig(short_prompt_threshold=10000)
        engine = Offload(config=config)
        local = Node("local", max_memory_bytes=100)
        for i in range(5):
            from tests.membrane.test_origin_node import make_fragment

            f = make_fragment(str(i), size=20)
            local.store(f, is_primary=True)
        remote = Node("remote")
        decision = engine.decide(list(range(100)), local, [remote])
        assert not decision.local_compute

    def test_no_candidates_falls_back_local(self):
        engine = Offload()
        local = Node("local")
        decision = engine.decide(list(range(2048)), local, [])
        assert decision.local_compute
        assert "no candidate" in decision.reason

    def test_decision_has_estimated_cost(self):
        engine = Offload()
        local = Node("local")
        decision = engine.decide(list(range(10)), local, [])
        assert decision.estimated_cost_seconds >= 0.0

    def test_custom_cost_model_used(self):
        model = CostModel(compute_scale=2.0)
        engine = Offload(cost_model=model)
        assert engine.cost_model.compute_scale == 2.0

    def test_cost_model_affects_estimated_cost(self):
        """Higher compute_scale should yield higher estimated cost."""
        local = Node("local")
        tokens = list(range(1000))
        engine_fast = Offload(cost_model=CostModel(compute_scale=1.0))
        engine_slow = Offload(cost_model=CostModel(compute_scale=3.0))
        decision_fast = engine_fast.decide(tokens, local, [])
        decision_slow = engine_slow.decide(tokens, local, [])
        assert decision_slow.estimated_cost_seconds > decision_fast.estimated_cost_seconds

    def test_offload_reason_is_descriptive(self):
        engine = Offload()
        local = Node("local")
        remote = Node("remote")
        decision = engine.decide(list(range(2048)), local, [remote])
        assert decision.reason
        assert isinstance(decision.reason, str)
