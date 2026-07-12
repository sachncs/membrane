"""FragmentGraph: nodes = fragments, edges = typed relationships.

This module defines :class:`FragmentGraph`, the structural graph
that backs Membrane's relationship reasoning. Nodes are fragments
keyed by ``content_hash``; edges are typed relationships
(``"co_access"``, ``"semantic"``, ``"positional"``, etc.) recorded
per source/target/type triple.

The graph stores adjacency as a nested mapping
``adjacency[source][edge_type] -> set[target]`` so that:

* Type-filtered neighbor queries are O(1) on the type-keyed set.
* Aggregated neighbor queries (across types) deduplicate via
  set-union.
* Adding an edge never requires touching unrelated nodes.

Typical use cases:
    * Tracking co-access relationships between frequently reused
      fragments.
    * Representing semantic similarity edges for graph-based
      retrieval.
    * Encoding positional adjacency between fragments in a
      sequential context.

Thread safety:
    The class is **not thread-safe**. Provide external
    synchronization when sharing across threads.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment


class FragmentGraph:
    """Directed typed graph over fragments.

    Nodes are keyed by ``content_hash``. Edges are keyed by
    ``(source, target, type)``. Adjacency is stored per node per
    edge type for fast traversal.

    Attributes:
        nodes: Mapping from ``content_hash`` to the corresponding
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

        Existing entries with the same ``content_hash`` are
        overwritten. The node's adjacency dict is initialized so
        that subsequent :meth:`add_edge` calls do not need to
        guard against missing keys.

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
        # Aggregate across all edge types, deduplicating.
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
