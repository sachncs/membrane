"""Graph: typed fragment graph with prefetch and eviction hints.

This module defines two classes:

* :class:`Graph` — the structural data structure (nodes are
  fragments keyed by ``content_hash``; edges are typed
  relationships). It also exposes the prefetch and eviction-hint
  policies that higher-level components need.
* :class:`SubgraphRetrieval` — bounded BFS over a
  :class:`~membrane.weighted.Weighted` for retrieving connected
  components.

The :class:`Graph` stores adjacency as a nested mapping
``adjacency[source][edge_type] -> set[target]`` so that:

* Type-filtered neighbor queries are O(1) on the type-keyed set.
* Aggregated neighbor queries (across types) deduplicate via
  set-union.
* Adding an edge never requires touching unrelated nodes.

Thread safety:
    None of the classes are thread-safe. Provide external
    synchronization when sharing across threads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from membrane.fragment import Fragment

if TYPE_CHECKING:
    from membrane.weighted import Weighted

logger = logging.getLogger(__name__)


class Graph:
    """Directed typed graph over fragments with prefetch/eviction hints.

    Nodes are keyed by ``payload_hash`` (``identity.payload_hash``).
    Edges are keyed by ``(source, target, type)``. Adjacency is
    stored per node per edge type for fast traversal.

    The graph also owns the prefetch and eviction policy helpers
    that previously lived in :class:`GraphManager`. They are pure
    functions of the adjacency structure, so keeping them on the
    graph avoids the thin-wrapper class.

    Attributes:
        nodes: Mapping from ``payload_hash`` to the corresponding
            :class:`~membrane.fragment.Fragment`. Allows callers to
            recover fragment metadata from a node reference.
        adjacency: Nested mapping
            ``adjacency[source][edge_type] -> set[target]``. Stored
            per source for O(1) type-filtered neighbor lookups.
    """

    def __init__(self) -> None:
        """Initialize an empty fragment graph."""
        self.nodes: dict[str, Fragment] = {}
        self.adjacency: dict[str, dict[str, set[str]]] = {}

    def add_node(self, fragment: Fragment) -> None:
        """Add a fragment as a graph node.

        Existing entries with the same ``payload_hash`` are
        overwritten. The node's adjacency dict is initialized so
        that subsequent :meth:`add_edge` calls do not need to
        guard against missing keys.

        Args:
            fragment: Fragment to add.
        """
        self.nodes[fragment.identity.payload_hash] = fragment
        self.adjacency.setdefault(fragment.identity.payload_hash, {})

    def add_edge(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
    ) -> None:
        """Add a typed edge between two fragment hashes.

        The edge is *directed*: ``(source, target)`` is recorded,
        but the reverse direction is not implied. Callers that
        need undirected semantics must add the reverse edge
        explicitly.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Type of relationship
                (e.g., ``"co_access"``, ``"semantic"``,
                ``"positional"``).
        """
        self.adjacency.setdefault(source_hash, {}).setdefault(edge_type, set()).add(target_hash)

    def has_node(self, content_hash: str) -> bool:
        """Check if a fragment hash is a node in the graph.

        Args:
            content_hash: Hash of the node to test.

        Returns:
            bool: True if the node has been added.
        """
        return content_hash in self.nodes

    def has_edge(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
    ) -> bool:
        """Check if a typed edge exists.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Type of relationship.

        Returns:
            bool: True if the edge has been added.
        """
        return target_hash in self.adjacency.get(source_hash, {}).get(edge_type, set())

    def neighbors(
        self,
        content_hash: str,
        edge_type: str | None = None,
    ) -> set[str]:
        """Return neighbors of a node.

        Args:
            content_hash: Hash of the node.
            edge_type: If provided, only neighbors reachable via
                this edge type are returned. Otherwise neighbors
                across all edge types are unioned.

        Returns:
            set[str]: Neighbor hashes. Empty when the node has no
            outgoing edges of the requested type.
        """
        types = self.adjacency.get(content_hash, {})
        if edge_type is not None:
            return set(types.get(edge_type, set()))
        result: set[str] = set()
        for neighbors in types.values():
            result.update(neighbors)
        return result

    def get_fragment(self, content_hash: str) -> Fragment | None:
        """Return the fragment associated with a node hash.

        Args:
            content_hash: Hash of the node to look up.

        Returns:
            Fragment | None: The fragment, or ``None`` if no node
            with that hash exists.
        """
        return self.nodes.get(content_hash)

    def prefetch_suggest(
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
        neighbors = self.neighbors(content_hash, edge_type)
        return list(neighbors)[:limit]

    def eviction_neighbors(
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
        return self.neighbors(content_hash, edge_type)


class SubgraphRetrieval:
    """Bounded BFS over a :class:`~membrane.weighted.Weighted`.

    Performs a small traversal on top of a weighted graph. Given a
    seed node, :meth:`retrieve_component` returns the set of hashes
    reachable within ``max_depth`` hops following edges whose
    weight meets or exceeds ``min_weight``.

    The retrieval is *weakly connected* in the sense that it follows
    edges in both directions implicitly (because
    :meth:`Weighted.get_strong_neighbors` already aggregates over
    all edge types originating at a node). For directed analyses
    where direction matters, build a one-sided weighted graph first.

    The class is a thin façade; all state lives in the supplied
    :class:`~membrane.weighted.Weighted`, so instances are safe to
    share across threads as long as the underlying graph itself
    is.

    Complexity:
        * :meth:`retrieve_component` — O(b^d) where ``b`` is the
          average branching factor and ``d`` is ``max_depth``.
        * :meth:`retrieve_clusters` — O(s · b^d) where ``s`` is the
          number of seeds, with deduplication so each node is
          visited at most once across the whole batch.
    """

    def __init__(self, graph: Weighted) -> None:
        """Initialize with a weighted graph.

        Args:
            graph: Graph to traverse. The instance is held by
                reference; mutations to it are visible to the
                retriever.
        """
        self.graph = graph

    def retrieve_component(
        self,
        seed_hash: str,
        min_weight: float = 0.5,
        max_depth: int = 3,
    ) -> set[str]:
        """Retrieve a connected component around ``seed_hash``.

        Performs a bounded breadth-first traversal starting from
        ``seed_hash``. At each level, every neighbor with weight
        above ``min_weight`` is added to the visited set and
        becomes part of the next frontier.

        Args:
            seed_hash: Starting fragment hash. If the graph does
                not contain the seed, the result is empty.
            min_weight: Minimum edge weight to follow (inclusive).
            max_depth: Maximum BFS depth. ``0`` returns just the
                seed; ``1`` adds the seed's strong neighbors, etc.

        Returns:
            set[str]: Fragment hashes in the component, including
            the seed.
        """
        if not self.graph.has_node(seed_hash):
            return set()

        visited: set[str] = {seed_hash}
        frontier: set[str] = {seed_hash}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                neighbors = self.graph.get_strong_neighbors(node, min_weight=min_weight)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return visited

    def retrieve_clusters(
        self,
        seed_hashes: list[str],
        min_weight: float = 0.5,
        max_depth: int = 3,
    ) -> list[set[str]]:
        """Retrieve clusters for multiple seeds with deduplication.

        Walks the seed list in order, skipping any seed that has
        already been visited by a previous cluster's BFS. The
        resulting list therefore contains *disjoint* sets.

        Args:
            seed_hashes: List of starting fragment hashes.
            min_weight: Minimum edge weight to follow (inclusive).
            max_depth: Maximum BFS depth per seed.

        Returns:
            list[set[str]]: One component per distinct seed, in
            input order. Components are mutually disjoint.
        """
        clusters: list[set[str]] = []
        seen: set[str] = set()
        for seed in seed_hashes:
            if seed in seen:
                continue
            component = self.retrieve_component(seed, min_weight, max_depth)
            clusters.append(component)
            seen.update(component)
        return clusters


__all__ = ["Graph", "SubgraphRetrieval"]
