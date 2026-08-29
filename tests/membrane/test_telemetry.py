from tests.conftest import make_fragment

"""Tests for telemetry module."""

import pytest

from membrane.node import Node
from membrane.telemetry import Telemetry


def test_collect_returns_telemetry():
    """Telemetry dataclass exposes the expected fields."""
    telem = Telemetry(
        node_id="n1",
        latency_ms=10.0,
        bandwidth_cost=0.5,
        gpu_load=0.3,
        memory_pressure=0.0,
    )
    assert telem.node_id == "n1"
    assert telem.latency_ms == 10.0
    assert telem.bandwidth_cost == 0.5
    assert telem.gpu_load == 0.3
    assert telem.memory_pressure == 0.0


def test_collect_memory_pressure_with_load():
    """memory_pressure reflects node heartbeat."""
    node = Node("n1", max_memory_bytes=100)

    f = make_fragment("x", size=50)
    node.store(f, is_primary=True)
    telem = Telemetry(
        node_id=node.node_id,
        latency_ms=0.0,
        bandwidth_cost=0.0,
        gpu_load=0.0,
        memory_pressure=node.heartbeat(),
    )
    assert telem.memory_pressure == 0.5


def test_default_values():
    """Defaults are zero."""
    telem = Telemetry(
        node_id="n1",
        latency_ms=0.0,
        bandwidth_cost=0.0,
        gpu_load=0.0,
        memory_pressure=0.0,
    )
    assert telem.memory_pressure == 0.0


def test_telemetry_immutable():
    """Telemetry is frozen."""
    telem = Telemetry(
        node_id="n1",
        latency_ms=1.0,
        bandwidth_cost=0.0,
        gpu_load=0.0,
        memory_pressure=0.0,
    )
    with pytest.raises(AttributeError):
        telem.latency_ms = 2.0
