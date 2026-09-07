from tests.conftest import make_fragment

"""Tests for joint_optimizer module."""

import pytest

from membrane.fragment import Fragment
from membrane.joint import Joint, PlacementDecision
from membrane.node import Node
from membrane.telemetry import Telemetry


class TestJointOptimizer:
    """Test suite for Joint."""

    def test_empty_nodes(self):
        opt = Joint()
        frag = make_fragment()
        decision = opt.optimize(frag, [], {})
        assert decision == PlacementDecision("", "", 0.0)

    def test_selects_lowest_gpu_compute(self):
        opt = Joint()
        n1 = Node("n1", max_memory_bytes=100)
        n2 = Node("n2", max_memory_bytes=100)

        f = make_fragment("x", size=80)
        n1.store(f, is_primary=True)
        telemetry = {
            "n1": Telemetry("n1", 10.0, 0.0, 0.9, 0.9),
            "n2": Telemetry("n2", 10.0, 0.0, 0.1, 0.1),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.compute_node_id == "n2"

    def test_splits_when_same_node_overloaded(self):
        opt = Joint()
        n1 = Node("n1", max_memory_bytes=100)
        n2 = Node("n2", max_memory_bytes=100)

        f = make_fragment("x", size=90)
        n1.store(f, is_primary=True)
        telemetry = {
            "n1": Telemetry("n1", 10.0, 0.0, 0.9, 0.9),
            "n2": Telemetry("n2", 10.0, 0.0, 0.5, 0.1),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.compute_node_id == "n2"
        assert decision.memory_node_id == "n2"

    def test_memory_node_lowest_pressure(self):
        opt = Joint()
        n1 = Node("n1")
        n2 = Node("n2")
        telemetry = {
            "n1": Telemetry("n1", 10.0, 0.0, 0.0, 0.9),
            "n2": Telemetry("n2", 10.0, 0.0, 0.0, 0.1),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.memory_node_id == "n2"

    def test_estimated_latency_from_telemetry(self):
        opt = Joint()
        n1 = Node("n1")
        telemetry = {
            "n1": Telemetry("n1", 500.0, 0.0, 0.0, 0.0),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1], telemetry)
        assert decision.estimated_latency_seconds == pytest.approx(0.5)

    def test_missing_telemetry_inf_score(self):
        opt = Joint()
        n1 = Node("n1")
        n2 = Node("n2")
        telemetry = {"n2": Telemetry("n2", 10.0, 0.0, 0.1, 0.1)}
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.compute_node_id == "n2"
        assert decision.memory_node_id == "n2"
