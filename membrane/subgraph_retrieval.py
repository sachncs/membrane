"""SubgraphRetrieval: retrieve connected components around a seed fragment."""

import logging

logger = logging.getLogger(__name__)


from membrane.weighted_graph import WeightedGraph


class SubgraphRetrieval:
    """Retrieves connected components from a weighted graph."""

    def __init__(self, graph: WeightedGraph) -> None:
        """Initialize with a weighted graph.

        Args:
            graph: Graph to traverse.
        """
        """Initialize with a weighted graph.

        Args:
            graph: Graph to traverse.
        """
        self.graph = graph

    def retrieve_component(
        self,
        seed_hash: str,
        min_weight: float = 0.5,
        max_depth: int = 3,
    ) -> set[str]:
        """Retrieve a connected component around a seed fragment.

        Args:
            seed_hash: Starting fragment hash.
            min_weight: Minimum edge weight to follow.
            max_depth: Maximum BFS depth.

        Returns:
            Set of fragment hashes in the component.
        """
        if not self.graph.has_node(seed_hash):
            return set()

        visited: set[str] = {seed_hash}
        frontier: set[str] = {seed_hash}

        for depth in range(max_depth):
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
        """Retrieve clusters for multiple seeds.

        Args:
            seed_hashes: List of starting fragment hashes.
            min_weight: Minimum edge weight to follow.
            max_depth: Maximum BFS depth.

        Returns:
            List of component sets.
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
