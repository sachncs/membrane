"""GraphManager: maintains the fragment relationship graph."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.fragment_graph import FragmentGraph


class GraphManager:
    """Manages the fragment graph for prefetch, replication, and eviction."""

    def __init__(self) -> None:
        logger.info("Initialized %s", self.__class__.__name__)
        self.graph = FragmentGraph()

    def register(self, fragment: Fragment) -> None:
        """Register a fragment in the graph.

        Args:
            fragment: Fragment to add as a node.
        """
        self.graph.add_node(fragment)

    def link(self, source_hash: str, target_hash: str, edge_type: str) -> None:
        """Create a typed relationship between two fragments.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Relationship type (e.g., "co_access", "semantic", "positional").
        """
        self.graph.add_edge(source_hash, target_hash, edge_type)

    def suggest_prefetch(
        self,
        content_hash: str,
        edge_type: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Suggest fragments to prefetch based on graph neighbors.

        Args:
            content_hash: Hash of the accessed fragment.
            edge_type: Optional edge type filter.
            limit: Maximum number of suggestions.

        Returns:
            List of suggested neighbor hashes.
        """
        neighbors = self.graph.neighbors(content_hash, edge_type)
        return list(neighbors)[:limit]

    def eviction_candidates(
        self,
        content_hash: str,
        edge_type: str | None = None,
    ) -> set[str]:
        """Return neighbor hashes that may also be cold if this node is evicted.

        Args:
            content_hash: Hash of the fragment being considered for eviction.
            edge_type: Optional edge type filter.

        Returns:
            Set of neighbor hashes for graph-aware eviction.
        """
        return self.graph.neighbors(content_hash, edge_type)
