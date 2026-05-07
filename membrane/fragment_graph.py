"""FragmentGraph: nodes = fragments, edges = typed relationships."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment


class FragmentGraph:
    """Directed typed graph over fragments.

    Nodes are keyed by content_hash. Edges are keyed by (source, target, type).
    Adjacency is stored per node per edge type for fast traversal.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Fragment] = {}
        self.adjacency: dict[str, dict[str, set[str]]] = {}

    def add_node(self, fragment: Fragment) -> None:
        """Add a fragment as a graph node.

        Args:
            fragment: Fragment to add.
        """
        self.nodes[fragment.content_hash] = fragment
        self.adjacency.setdefault(fragment.content_hash, {})

    def add_edge(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
    ) -> None:
        """Add a typed edge between two fragment hashes.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Type of relationship (e.g., "co_access", "semantic", "positional").
        """
        self.adjacency.setdefault(source_hash, {}).setdefault(edge_type, set()).add(
            target_hash
        )

    def has_node(self, content_hash: str) -> bool:
        """Check if a fragment hash is a node in the graph."""
        return content_hash in self.nodes

    def has_edge(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
    ) -> bool:
        """Check if a typed edge exists."""
        return target_hash in self.adjacency.get(source_hash, {}).get(edge_type, set())

    def neighbors(
        self,
        content_hash: str,
        edge_type: str | None = None,
    ) -> set[str]:
        """Return neighbors of a node.

        Args:
            content_hash: Hash of the node.
            edge_type: If provided, filter by edge type.

        Returns:
            Set of neighbor hashes.
        """
        types = self.adjacency.get(content_hash, {})
        if edge_type is not None:
            return set(types.get(edge_type, set()))
        result: set[str] = set()
        for neighbors in types.values():
            result.update(neighbors)
        return result

    def get_fragment(self, content_hash: str) -> Fragment | None:
        """Return the Fragment associated with a node hash."""
        return self.nodes.get(content_hash)
