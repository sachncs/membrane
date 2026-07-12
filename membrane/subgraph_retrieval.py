"""SubgraphRetrieval: retrieve connected components around a seed fragment.

This module implements a small BFS-style traversal on top of
:class:`~membrane.weighted_graph.WeightedGraph`. Given a seed node,
:meth:`retrieve_component` returns the set of hashes reachable
within ``max_depth`` hops following edges whose weight meets or
exceeds ``min_weight``.

The retrieval is *weakly connected* in the sense that it follows
edges in both directions implicitly (because
:meth:`WeightedGraph.get_strong_neighbors` already aggregates over
all edge types originating at a node). For directed analyses
where direction matters, build a one-sided weighted graph first.

Complexity:
    * :meth:`retrieve_component` — O(b^d) where ``b`` is the
      average branching factor and ``d`` is ``max_depth``.
    * :meth:`retrieve_clusters` — O(s · b^d) where ``s`` is the
      number of seeds, with deduplication so each node is visited
      at most once across the whole batch.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.weighted_graph import WeightedGraph


class SubgraphRetrieval:
    """Retrieves connected components from a weighted graph.

    The class is a thin façade; all state lives in the supplied
    :class:`~membrane.weighted_graph.WeightedGraph`, so instances
    are safe to share across threads as long as the underlying
    graph itself is.
    """

    def __init__(self, graph: WeightedGraph) -> None:
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
            # next_frontier collects every node reached in this
            # BFS layer.
            next_frontier: set[str] = set()
            for node in frontier:
                neighbors = self.graph.get_strong_neighbors(node, min_weight=min_weight)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                # No new nodes reached; the component is fully
                # explored and we can stop early.
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
                # Already covered by a prior cluster.
                continue
            component = self.retrieve_component(seed, min_weight, max_depth)
            clusters.append(component)
            # Mark every node in this component so future seeds
            # that fall inside it are skipped.
            seen.update(component)
        return clusters
