"""LatencyRouter: route requests by latency to local, replica, or origin.

This module defines :class:`LatencyRouter`, a request-time router
that picks the lowest-latency node holding a requested fragment.

The priority order is:

1. **Local node** — if the local node already has the fragment,
   the lookup is served directly (zero network hop).
2. **Best replica** — among the candidates that hold the
   fragment, the one with the lowest recorded latency is chosen.
3. **Origin fallback** — when no candidate holds the fragment,
   the router falls back to a configured origin (or the local
   node id if no origin is configured).

The router is intentionally stateless apart from its
``latency_table``. Callers update the table as new measurements
become available via :meth:`add_latency`.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode


class LatencyRouter:
    """Routes fragment lookups based on latency tiers.

    Priority:
        1. Local node exact match
        2. Nearest replica with lowest latency
        3. Origin node (fallback)

    Attributes:
        latency_table: Mapping ``node_id -> latency_ms`` used to
            score replica candidates.
        origin_node_id: Optional fallback node id used when no
            candidate holds the fragment. ``None`` causes the
            local node id to be used instead.
    """

    def __init__(
        self,
        latency_table: dict[str, float] | None = None,
        origin_node_id: str | None = None,
    ) -> None:
        """Initialize with optional latency table and origin fallback.

        Args:
            latency_table: Mapping of ``node_id -> latency`` in
                milliseconds.
            origin_node_id: Node ID to use as fallback when no
                replica holds the fragment. ``None`` causes the
                router to fall back to ``local_node.node_id``.
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
            candidate_nodes: Other nodes that may hold the
                fragment.

        Returns:
            str: Selected node identifier. Always one of the
            ``node_id`` values from ``local_node`` or
            ``candidate_nodes``, or the configured origin id.
        """
        # Fast path: local exact match.
        if local_node.retrieve(content_hash) is not None:
            return local_node.node_id

        # Replica path: filter to candidates that actually hold
        # the fragment, then pick the one with the lowest
        # recorded latency.
        candidates_with_fragment = [
            node
            for node in candidate_nodes
            if node.retrieve(content_hash) is not None
        ]

        if not candidates_with_fragment:
            # Fallback: origin if configured, otherwise local.
            fallback = self.origin_node_id or local_node.node_id
            logger.debug(
                "No replica for %s; falling back to %s", content_hash, fallback
            )
            return fallback

        def latency_key(node: MembraneNode) -> float:
            """Latency score (lower is better); infinity if unknown."""
            return self.latency_table.get(node.node_id, float("inf"))

        best = min(candidates_with_fragment, key=latency_key)
        return best.node_id

    def get_latency(self, node_id: str) -> float:
        """Return recorded latency for a node.

        Args:
            node_id: Node identifier.

        Returns:
            float: Latency in milliseconds, or ``inf`` if the
            node is not in the table.
        """
        return self.latency_table.get(node_id, float("inf"))
