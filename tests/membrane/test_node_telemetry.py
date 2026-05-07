"""Tests for node_telemetry module."""

import pytest

from membrane.membrane_node import MembraneNode
from membrane.node_telemetry import NodeTelemetry, TelemetryCollector


class TestTelemetryCollector:
    """Test suite for TelemetryCollector."""

    def test_collect_returns_telemetry(self):
        tc = TelemetryCollector()
        node = MembraneNode("n1")
        telem = tc.collect(node, latency_ms=10.0, bandwidth_cost=0.5, gpu_load=0.3)
        assert isinstance(telem, NodeTelemetry)
        assert telem.node_id == "n1"
        assert telem.latency_ms == 10.0
        assert telem.bandwidth_cost == 0.5
        assert telem.gpu_load == 0.3
        assert telem.memory_pressure == 0.0

    def test_collect_memory_pressure_with_load(self):
        tc = TelemetryCollector()
        node = MembraneNode("n1", max_memory_bytes=100)
        from tests.membrane.test_cluster_replicator import make_fragment

        f = make_fragment("x", size=50)
        node.store(f, is_primary=True)
        telem = tc.collect(node)
        assert telem.memory_pressure == 0.5

    def test_default_values(self):
        tc = TelemetryCollector()
        node = MembraneNode("n1")
        telem = tc.collect(node)
        assert telem.latency_ms == 0.0
        assert telem.bandwidth_cost == 0.0
        assert telem.gpu_load == 0.0

    def test_telemetry_immutable(self):
        telem = NodeTelemetry(
            node_id="n1",
            latency_ms=1.0,
            bandwidth_cost=0.0,
            gpu_load=0.0,
            memory_pressure=0.0,
        )
        with pytest.raises(AttributeError):
            telem.latency_ms = 2.0
