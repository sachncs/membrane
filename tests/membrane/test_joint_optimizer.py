"""Tests for joint_optimizer module."""

import pytest

from membrane.fragment import Fragment
from membrane.joint_optimizer import JointOptimizer, PlacementDecision
from membrane.membrane_node import MembraneNode
from membrane.node_telemetry import NodeTelemetry
from membrane.structural_signature import StructuralSignature


def make_fragment():
    return Fragment(
        content_hash="abc",
        embedding=(0.0,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=10,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestJointOptimizer:
    """Test suite for JointOptimizer."""

    def test_empty_nodes(self):
        opt = JointOptimizer()
        frag = make_fragment()
        decision = opt.optimize(frag, [], {})
        assert decision == PlacementDecision("", "", 0.0)

    def test_selects_lowest_gpu_compute(self):
        opt = JointOptimizer()
        n1 = MembraneNode("n1", max_memory_bytes=100)
        n2 = MembraneNode("n2", max_memory_bytes=100)
        from tests.membrane.test_cluster_replicator import make_fragment as mkfrag

        f = mkfrag("x", size=80)
        n1.store(f, is_primary=True)
        telemetry = {
            "n1": NodeTelemetry("n1", 10.0, 0.0, 0.9, 0.9),
            "n2": NodeTelemetry("n2", 10.0, 0.0, 0.1, 0.1),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.compute_node_id == "n2"

    def test_splits_when_same_node_overloaded(self):
        opt = JointOptimizer()
        n1 = MembraneNode("n1", max_memory_bytes=100)
        n2 = MembraneNode("n2", max_memory_bytes=100)
        from tests.membrane.test_cluster_replicator import make_fragment as mkfrag

        f = mkfrag("x", size=90)
        n1.store(f, is_primary=True)
        telemetry = {
            "n1": NodeTelemetry("n1", 10.0, 0.0, 0.9, 0.9),
            "n2": NodeTelemetry("n2", 10.0, 0.0, 0.5, 0.1),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.compute_node_id == "n2"
        assert decision.memory_node_id == "n2"

    def test_memory_node_lowest_pressure(self):
        opt = JointOptimizer()
        n1 = MembraneNode("n1")
        n2 = MembraneNode("n2")
        telemetry = {
            "n1": NodeTelemetry("n1", 10.0, 0.0, 0.0, 0.9),
            "n2": NodeTelemetry("n2", 10.0, 0.0, 0.0, 0.1),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.memory_node_id == "n2"

    def test_estimated_latency_from_telemetry(self):
        opt = JointOptimizer()
        n1 = MembraneNode("n1")
        telemetry = {
            "n1": NodeTelemetry("n1", 500.0, 0.0, 0.0, 0.0),
        }
        frag = make_fragment()
        decision = opt.optimize(frag, [n1], telemetry)
        assert decision.estimated_latency_seconds == pytest.approx(0.5)

    def test_missing_telemetry_inf_score(self):
        opt = JointOptimizer()
        n1 = MembraneNode("n1")
        n2 = MembraneNode("n2")
        telemetry = {"n2": NodeTelemetry("n2", 10.0, 0.0, 0.1, 0.1)}
        frag = make_fragment()
        decision = opt.optimize(frag, [n1, n2], telemetry)
        assert decision.compute_node_id == "n2"
        assert decision.memory_node_id == "n2"
