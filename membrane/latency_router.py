"""LatencyRouter: route requests by latency to local, replica, or origin."""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode


class LatencyRouter:
    """Routes fragment lookups based on latency tiers.

    Priority:
    1. Local node exact match
    2. Nearest replica with lowest latency
    3. Origin node (fallback)
    """

    def __init__(
        self,
        latency_table: dict[str, float] | None = None,
        origin_node_id: str | None = None,
    ) -> None:
        """Initialize with optional latency table and origin fallback.

        Args:
            latency_table: Mapping of node_id -> latency in milliseconds.
            origin_node_id: Node ID to use as fallback when no replica holds
                the fragment. If None, falls back to local_node.node_id.
        """
        self.latency_table: dict[str, float] = latency_table or {}
        self.origin_node_id = origin_node_id

    def add_latency(self, node_id: str, latency_ms: float) -> None:
        """Record latency to a node.

        Args:
            node_id: Target node identifier.
            latency_ms: Round-trip latency in milliseconds.
        """
        self.latency_table[node_id] = latency_ms

    def route_local_or_replica(
        self,
        content_hash: str,
        local_node: MembraneNode,
        candidate_nodes: list[MembraneNode],
    ) -> str:
        """Select the best node to serve a fragment lookup.

        Args:
            content_hash: Fragment hash to retrieve.
            local_node: Node processing the request.
            candidate_nodes: Other nodes that may hold the fragment.

        Returns:
            Selected node identifier.
        """
        if local_node.retrieve(content_hash) is not None:
            return local_node.node_id

        candidates_with_fragment = [
            node
            for node in candidate_nodes
            if node.retrieve(content_hash) is not None
        ]

        if not candidates_with_fragment:
            # Fallback to origin node if configured, otherwise local node
            fallback = self.origin_node_id or local_node.node_id
            logger.debug(
                "No replica for %s; falling back to %s", content_hash, fallback
            )
            return fallback

        def latency_key(node: MembraneNode) -> float:
            """Return latency for a candidate node."""
            return self.latency_table.get(node.node_id, float("inf"))

        best = min(candidates_with_fragment, key=latency_key)
        return best.node_id

    def get_latency(self, node_id: str) -> float:
        """Return recorded latency for a node.

        Args:
            node_id: Node identifier.

        Returns:
            Latency in milliseconds, or infinity if unknown.
        """
        return self.latency_table.get(node_id, float("inf"))
