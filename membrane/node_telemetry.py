"""NodeTelemetry: latency, bandwidth cost, and GPU load reporting.

This module defines :class:`NodeTelemetry` (a frozen telemetry
snapshot) and :class:`TelemetryCollector` (a small helper that
produces snapshots from a :class:`~membrane.membrane_node
.MembraneNode`).

The snapshot is the unit of information the routing and decision
layers (e.g., :class:`~membrane.economic_router.EconomicRouter`,
:class:`~membrane.joint_optimizer.JointOptimizer`) consume. By
keeping the snapshot *immutable* the routing code can rely on it
being a coherent, untampered view of the node at a specific
moment, even if the underlying node continues to mutate.

Thread safety:
    Both classes are stateless and safe to share across threads.
"""

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
        bandwidth_cost: Relative bandwidth cost unit
            (``0.0`` = free, ``1.0`` = expensive).
        gpu_load: GPU utilization ratio in ``[0.0, 1.0]``.
        memory_pressure: Memory usage ratio in ``[0.0, 1.0]``,
            typically derived from the node's
            :meth:`~MembraneNode.heartbeat` at snapshot time.
    """

    node_id: str
    latency_ms: float
    bandwidth_cost: float
    gpu_load: float
    memory_pressure: float


class TelemetryCollector:
    """Collects telemetry from :class:`MembraneNode` instances."""

    def __init__(self) -> None:
        """Initialize the collector."""
        pass

    def collect(
        self,
        node: MembraneNode,
        latency_ms: float = 0.0,
        bandwidth_cost: float = 0.0,
        gpu_load: float = 0.0,
    ) -> NodeTelemetry:
        """Collect a telemetry snapshot from ``node``.

        Memory pressure is sourced from the node itself via
        :meth:`MembraneNode.heartbeat`; the other three
        dimensions are caller-supplied because they cannot be
        observed from the in-process node alone.

        Args:
            node: Node to inspect.
            latency_ms: Measured or estimated latency.
            bandwidth_cost: Relative bandwidth cost.
            gpu_load: GPU utilization.

        Returns:
            NodeTelemetry: Frozen snapshot.
        """
        return NodeTelemetry(
            node_id=node.node_id,
            latency_ms=latency_ms,
            bandwidth_cost=bandwidth_cost,
            gpu_load=gpu_load,
            # Heart beat already normalizes memory usage into
            # [0, 1], matching the dataclass contract.
            memory_pressure=node.heartbeat(),
        )
