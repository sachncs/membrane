"""telemetry: latency, bandwidth cost, and GPU load reporting.

This module defines :class:`Telemetry`, a frozen telemetry snapshot
consumed by the routing and decision layers. By keeping the snapshot
immutable, routing code can rely on it being a coherent view of the
node at a specific moment even if the underlying node continues to
mutate.

Thread safety:
    The dataclass is stateless and safe to share across threads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Telemetry:
    """Telemetry snapshot for a single node.

    Attributes:
        node_id: Node identifier.
        latency_ms: Average response latency in milliseconds.
        bandwidth_cost: Relative bandwidth cost unit
            (``0.0`` = free, ``1.0`` = expensive).
        gpu_load: GPU utilization ratio in ``[0.0, 1.0]``.
        memory_pressure: Memory usage ratio in ``[0.0, 1.0]``,
            typically derived from the node's
            :meth:`~membrane.node.Node.heartbeat` at snapshot time.
    """

    node_id: str
    latency_ms: float
    bandwidth_cost: float
    gpu_load: float
    memory_pressure: float


__all__ = ["Telemetry"]
