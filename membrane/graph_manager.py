"""GraphManager: maintains the fragment relationship graph.

This module defines :class:`GraphManager`, the application-level
wrapper around :class:`~membrane.fragment_graph.FragmentGraph`.
It exposes three operations that higher-level Membrane components
care about:

* **Registration** (:meth:`register`) — add a fragment as a graph
  node.
* **Linking** (:meth:`link`) — declare a typed relationship
  between two fragments.
* **Suggestion** (:meth:`suggest_prefetch`,
  :meth:`eviction_candidates`) — derive hints for prefetching and
  eviction from the graph.

The manager does not own the graph — it delegates every operation
to an internal :class:`~membrane.fragment_graph.FragmentGraph`.
This separation keeps the data structure reusable independently
of the manager's higher-level semantics.

Thread safety:
    The class inherits the non-thread-safe behavior of the
    underlying graph. Provide external locking when sharing across
    threads.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.fragment_graph import FragmentGraph


class GraphManager:
    """Manages the fragment graph for prefetch, replication, and eviction.

    Attributes:
        graph: The underlying
            :class:`~membrane.fragment_graph.FragmentGraph`. Held
            by reference so callers may iterate it directly when
            they need lower-level access.
    """

    def __init__(self) -> None:
        """Initialize the manager with an empty fragment graph."""
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

        The edge is directed. Endpoints do not need to exist in
        the graph ahead of time — the underlying
        :class:`~membrane.fragment_graph.FragmentGraph` will lazily
        create empty adjacency entries.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Relationship type
                (e.g., ``"co_access"``, ``"semantic"``,
                ``"positional"``).
        """
        self.graph.add_edge(source_hash, target_hash, edge_type)

    def suggest_prefetch(
        self,
        content_hash: str,
        edge_type: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Suggest fragments to prefetch based on graph neighbors.

        The implementation simply takes the first ``limit``
        neighbors; callers wanting stronger rankings should layer
        a custom scoring policy on top.

        Args:
            content_hash: Hash of the accessed fragment.
            edge_type: Optional edge type filter. When ``None``,
                neighbors across all edge types are considered.
            limit: Maximum number of suggestions to return.

        Returns:
            list[str]: Suggested neighbor hashes, ordered by
            iteration order of the underlying set.
        """
        neighbors = self.graph.neighbors(content_hash, edge_type)
        return list(neighbors)[:limit]

    def eviction_candidates(
        self,
        content_hash: str,
        edge_type: str | None = None,
    ) -> set[str]:
        """Return neighbor hashes that may also be cold if this node is evicted.

        This is a *hint* API — it returns the structural neighbors
        so the eviction policy can decide whether to evict them in
        tandem (e.g., for graph-aware compaction).

        Args:
            content_hash: Hash of the fragment being considered
                for eviction.
            edge_type: Optional edge type filter.

        Returns:
            set[str]: Neighbor hashes for graph-aware eviction.
        """
        return self.graph.neighbors(content_hash, edge_type)
