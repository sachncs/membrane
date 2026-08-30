"""Latency: route requests by latency to local, replica, or origin.

This module defines :class:`Latency`, a request-time router
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


from membrane.node import Node


class Latency:
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

    def get_latency(self, node_id: str) -> float:
        """Return recorded latency for a node.

        Args:
            node_id: Node identifier.

        Returns:
            float: Latency in milliseconds, or ``inf`` if the
            node is not in the table.
        """
        return self.latency_table.get(node_id, float("inf"))

    @staticmethod
    def _holds(node: Node, content_hash: str) -> bool:
        """Return ``True`` when ``node`` actually holds ``content_hash``.

        This is an idempotent probe; the call has the side effect
        of bumping the node's access timestamp.

        Args:
            node: Node to probe.
            content_hash: Hash to look up.

        Returns:
            bool: True when ``node`` returns a non-``None``
            fragment for the hash.
        """
        return node.retrieve(content_hash) is not None

    def _pick_local(self, content_hash: str, local_node: Node) -> str | None:
        """Return ``local_node.node_id`` iff it holds the fragment."""
        if self._holds(local_node, content_hash):
            return local_node.node_id
        return None

    def _pick_replica(
        self,
        content_hash: str,
        candidate_nodes: list[Node],
    ) -> Node | None:
        """Return the candidate with the lowest recorded latency, if any.

        Args:
            content_hash: Hash to look up.
            candidate_nodes: Candidate replicas.

        Returns:
            Node | None: The lowest-latency node that actually
            holds the fragment, or ``None`` when none do.
        """
        holding = [node for node in candidate_nodes if self._holds(node, content_hash)]
        if not holding:
            return None

        def latency_key(node: Node) -> float:
            """Latency score (lower is better); infinity if unknown."""
            return self.latency_table.get(node.node_id, float("inf"))

        return min(holding, key=latency_key)

    def _pick_fallback(
        self,
        local_node: Node,
    ) -> str:
        """Return the fallback node id used when no replica holds the fragment."""
        return self.origin_node_id or local_node.node_id

    def pick_target(
        self,
        content_hash: str,
        local_node: Node,
        candidate_nodes: list[Node],
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
        local_target = self._pick_local(content_hash, local_node)
        if local_target is not None:
            return local_target

        replica = self._pick_replica(content_hash, candidate_nodes)
        if replica is not None:
            return replica.node_id

        fallback = self._pick_fallback(local_node)
        logger.debug("No replica for %s; falling back to %s", content_hash, fallback)
        return fallback


__all__ = ["Latency"]
