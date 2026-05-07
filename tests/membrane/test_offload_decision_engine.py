"""Tests for offload_decision_engine module."""

import pytest

from membrane.cost_model import CostModel
from membrane.membrane_node import MembraneNode
from membrane.offload_decision_engine import (
    OffloadDecision,
    OffloadDecisionConfig,
    OffloadDecisionEngine,
)


class TestOffloadDecisionEngine:
    """Test suite for OffloadDecisionEngine."""

    def test_short_prompt_low_load_local(self):
        engine = OffloadDecisionEngine()
        local = MembraneNode("local")
        decision = engine.decide(list(range(100)), local, [])
        assert decision.local_compute
        assert decision.target_node_id == "local"
        assert "short" in decision.reason

    def test_long_prompt_offloads(self):
        engine = OffloadDecisionEngine()
        local = MembraneNode("local")
        remote = MembraneNode("remote")
        decision = engine.decide(list(range(2048)), local, [remote])
        assert not decision.local_compute
        assert decision.target_node_id == "remote"

    def test_high_local_load_offloads(self):
        config = OffloadDecisionConfig(short_prompt_threshold=10000)
        engine = OffloadDecisionEngine(config=config)
        local = MembraneNode("local", max_memory_bytes=100)
        for i in range(5):
            from tests.membrane.test_origin_node import make_fragment

            f = make_fragment(str(i), size=20)
            local.store(f, is_primary=True)
        remote = MembraneNode("remote")
        decision = engine.decide(list(range(100)), local, [remote])
        assert not decision.local_compute

    def test_no_candidates_falls_back_local(self):
        engine = OffloadDecisionEngine()
        local = MembraneNode("local")
        decision = engine.decide(list(range(2048)), local, [])
        assert decision.local_compute
        assert "no candidate" in decision.reason

    def test_decision_has_estimated_cost(self):
        engine = OffloadDecisionEngine()
        local = MembraneNode("local")
        decision = engine.decide(list(range(10)), local, [])
        assert decision.estimated_cost_seconds >= 0.0

    def test_custom_cost_model_used(self):
        model = CostModel(compute_scale=2.0)
        engine = OffloadDecisionEngine(cost_model=model)
        assert engine.cost_model.compute_scale == 2.0

    def test_cost_model_affects_estimated_cost(self):
        """Higher compute_scale should yield higher estimated cost."""
        local = MembraneNode("local")
        tokens = list(range(1000))
        engine_fast = OffloadDecisionEngine(cost_model=CostModel(compute_scale=1.0))
        engine_slow = OffloadDecisionEngine(cost_model=CostModel(compute_scale=3.0))
        decision_fast = engine_fast.decide(tokens, local, [])
        decision_slow = engine_slow.decide(tokens, local, [])
        assert decision_slow.estimated_cost_seconds > decision_fast.estimated_cost_seconds

    def test_offload_reason_is_descriptive(self):
        engine = OffloadDecisionEngine()
        local = MembraneNode("local")
        remote = MembraneNode("remote")
        decision = engine.decide(list(range(2048)), local, [remote])
        assert decision.reason
        assert isinstance(decision.reason, str)
