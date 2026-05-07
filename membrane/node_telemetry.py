"""NodeTelemetry: latency, bandwidth cost, and GPU load reporting."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.membrane_node import MembraneNode


@dataclass(frozen=True)
class NodeTelemetry:
    """Telemetry snapshot for a single node.

    Attributes:
        node_id: Node identifier.
        latency_ms: Average response latency in milliseconds.
        bandwidth_cost: Relative bandwidth cost unit (0.0 = free, 1.0 = expensive).
        gpu_load: GPU utilization ratio in [0.0, 1.0].
        memory_pressure: Memory usage ratio in [0.0, 1.0].
    """

    node_id: str
    latency_ms: float
    bandwidth_cost: float
    gpu_load: float
    memory_pressure: float


class TelemetryCollector:
    """Collects telemetry from MembraneNode instances."""

    def __init__(self) -> None:
        """Initialize the collector."""
        """Initialize the collector."""
        pass

    def collect(
        self,
        node: MembraneNode,
        latency_ms: float = 0.0,
        bandwidth_cost: float = 0.0,
        gpu_load: float = 0.0,
    ) -> NodeTelemetry:
        """Collect a telemetry snapshot from a node.

        Args:
            node: Node to inspect.
            latency_ms: Measured or estimated latency.
            bandwidth_cost: Relative bandwidth cost.
            gpu_load: GPU utilization.

        Returns:
            Telemetry snapshot.
        """
        return NodeTelemetry(
            node_id=node.node_id,
            latency_ms=latency_ms,
            bandwidth_cost=bandwidth_cost,
            gpu_load=gpu_load,
            memory_pressure=node.heartbeat(),
        )
