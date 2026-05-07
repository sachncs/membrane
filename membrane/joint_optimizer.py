"""JointOptimizer: jointly optimize memory placement and compute placement."""

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
        memory_node_id: Node to store resulting fragments.
        estimated_latency_seconds: Expected end-to-end latency.
    """

    compute_node_id: str
    memory_node_id: str
    estimated_latency_seconds: float


class JointOptimizer:
    """Optimizes both where to compute and where to store results."""

    def __init__(self) -> None:
        """Initialize the joint optimizer."""
        """Initialize the joint optimizer."""
        pass

    def optimize(
        self,
        fragment: Fragment,
        nodes: list[MembraneNode],
        telemetry_map: dict[str, NodeTelemetry],
    ) -> PlacementDecision:
        """Jointly select compute node and memory node.

        Strategy:
        - Compute node: lowest GPU load
        - Memory node: lowest memory pressure with existing adjacency

        Args:
            fragment: Fragment to place.
            nodes: Candidate nodes.
            telemetry_map: Telemetry for each node.

        Returns:
            PlacementDecision with compute and memory targets.
        """
        if not nodes:
            return PlacementDecision(
                compute_node_id="",
                memory_node_id="",
                estimated_latency_seconds=0.0,
            )

        # Compute node: minimize GPU load + latency
        def compute_score(node: MembraneNode) -> float:
            """Score compute suitability: lower is better."""
            telem = telemetry_map.get(node.node_id)
            if telem is None:
                return float("inf")
            return telem.gpu_load + telem.latency_ms / 1000.0

        compute_node = min(nodes, key=compute_score)

        # Memory node: minimize memory pressure
        def memory_score(node: MembraneNode) -> float:
            """Score memory suitability: lower is better."""
            telem = telemetry_map.get(node.node_id)
            if telem is None:
                return float("inf")
            return telem.memory_pressure

        memory_node = min(nodes, key=memory_score)

        # If compute and memory are the same overloaded node, split them
        if (
            compute_node.node_id == memory_node.node_id
            and compute_node.heartbeat() > 0.8
        ):
            alt_nodes = [n for n in nodes if n.node_id != compute_node.node_id]
            if alt_nodes:
                memory_node = min(alt_nodes, key=memory_score)

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
