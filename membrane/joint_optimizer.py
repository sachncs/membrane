"""JointOptimizer: jointly optimize memory placement and compute placement.

This module defines :class:`JointOptimizer` and its supporting
:class:`PlacementDecision` dataclass. The optimizer picks two
nodes for a fragment — one to *compute* (prefill or decode) and
one to *store* the resulting fragments — with the constraint that
the two may be the same node when the load allows, but should be
split when the joint node is too loaded.

Heuristic:

* **Compute node**: minimize ``gpu_load + latency_ms / 1000``.
* **Memory node**: minimize ``memory_pressure``; if the
  candidate coincides with the compute node and the node is
  heavily loaded (``heartbeat() > 0.8``), pick a different node
  for memory.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.node_telemetry import NodeTelemetry


@dataclass(frozen=True)
class PlacementDecision:
    """Joint placement decision for memory and compute.

    Attributes:
        compute_node_id: Node to perform prefill/decode.
        memory_node_id: Node to store resulting fragments. May be
            the same as ``compute_node_id`` when load permits.
        estimated_latency_seconds: Expected end-to-end latency
            derived from the compute node's telemetry.
    """

    compute_node_id: str
    memory_node_id: str
    estimated_latency_seconds: float


class JointOptimizer:
    """Optimizes both where to compute and where to store results.

    The optimizer is stateless; instances are safe to share
    across threads as long as the supplied ``MembraneNode``
    references are themselves safe to query.
    """

    def __init__(self) -> None:
        """Initialize the joint optimizer."""
        pass

    def optimize(
        self,
        fragment: Fragment,
        nodes: list[MembraneNode],
        telemetry_map: dict[str, NodeTelemetry],
    ) -> PlacementDecision:
        """Jointly select compute node and memory node.

        Args:
            fragment: Fragment to place. Currently unused by the
                scoring logic but accepted for forward
                compatibility with fragment-aware heuristics.
            nodes: Candidate nodes.
            telemetry_map: ``node_id -> NodeTelemetry`` snapshot.

        Returns:
            PlacementDecision: Selected compute and memory
            targets plus the estimated end-to-end latency. When
            ``nodes`` is empty, both targets are empty strings
            and the estimated latency is ``0.0``.
        """
        if not nodes:
            return PlacementDecision(
                compute_node_id="",
                memory_node_id="",
                estimated_latency_seconds=0.0,
            )

        # Compute node: minimize GPU load + (latency in seconds).
        # Adding latency in seconds (rather than milliseconds)
        # keeps both terms on the same order of magnitude for
        # typical GPU loads in the 0.1-1.0 range.
        def compute_score(node: MembraneNode) -> float:
            """Score compute suitability (lower is better)."""
            telem = telemetry_map.get(node.node_id)
            if telem is None:
                return float("inf")
            return telem.gpu_load + telem.latency_ms / 1000.0

        compute_node = min(nodes, key=compute_score)

        # Memory node: minimize memory pressure.
        def memory_score(node: MembraneNode) -> float:
            """Score memory suitability (lower is better)."""
            telem = telemetry_map.get(node.node_id)
            if telem is None:
                return float("inf")
            return telem.memory_pressure

        memory_node = min(nodes, key=memory_score)

        # If compute and memory coincide on a heavily-loaded
        # node, split them by picking the next-best memory node.
        if compute_node.node_id == memory_node.node_id and compute_node.heartbeat() > 0.8:
            alt_nodes = [n for n in nodes if n.node_id != compute_node.node_id]
            if alt_nodes:
                memory_node = min(alt_nodes, key=memory_score)

        # Estimated end-to-end latency is the compute node's
        # round-trip latency, converted to seconds. A zero-valued
        # fallback telemetry is used when the compute node has
        # no entry in the telemetry map.
        est_latency = (
            telemetry_map.get(
                compute_node.node_id,
                NodeTelemetry(
                    node_id=compute_node.node_id,
                    latency_ms=0.0,
                    bandwidth_cost=0.0,
                    gpu_load=0.0,
                    memory_pressure=0.0,
                ),
            ).latency_ms
            / 1000.0
        )

        return PlacementDecision(
            compute_node_id=compute_node.node_id,
            memory_node_id=memory_node.node_id,
            estimated_latency_seconds=est_latency,
        )
