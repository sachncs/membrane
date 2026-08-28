"""telemetry: latency, bandwidth cost, and GPU load reporting.

This module defines :class:`Telemetry` (a frozen telemetry snapshot)
and :func:`telemetry` (a free function that produces snapshots from a
:class:`~membrane.node.Node`).

The snapshot is the unit of information the routing and decision layers
consume. By keeping the snapshot immutable, routing code can rely on it
being a coherent view of the node at a specific moment even if the
underlying node continues to mutate.

Thread safety:
    Both the dataclass and the function are stateless and safe to
    share across threads.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.node import Node


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
            :meth:`~Node.heartbeat` at snapshot time.
    """

    node_id: str
    latency_ms: float
    bandwidth_cost: float
    gpu_load: float
    memory_pressure: float


def telemetry(
    node: Node,
    latency_ms: float = 0.0,
    bandwidth_cost: float = 0.0,
    gpu_load: float = 0.0,
) -> Telemetry:
    """Collect a telemetry snapshot from ``node``.

    Memory pressure is sourced from the node itself via
    :meth:`Node.heartbeat`; the other three dimensions are
    caller-supplied because they cannot be observed from the
    in-process node alone.

    Args:
        node: Node to inspect.
        latency_ms: Measured or estimated latency.
        bandwidth_cost: Relative bandwidth cost.
        gpu_load: GPU utilization.

    Returns:
        Telemetry: Frozen snapshot.
    """
    return Telemetry(
        node_id=node.node_id,
        latency_ms=latency_ms,
        bandwidth_cost=bandwidth_cost,
        gpu_load=gpu_load,
        memory_pressure=node.heartbeat(),
    )
