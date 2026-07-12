"""WeightedGraph: graph edges with co-occurrence probabilities.

This module defines :class:`WeightedGraph`, a thin wrapper around
:class:`~membrane.fragment_graph.FragmentGraph` that adds
**typed edges with float weights** on top of the plain structural
graph. Weights typically encode co-occurrence or reuse
probabilities, allowing downstream components to reason about
"how strongly" two fragments are related rather than merely
*whether* they are related.

The wrapper keeps the underlying structural graph intact and adds a
parallel ``weights`` dict indexed by ``(source, edge_type, target)``.
Nodes referenced only via :meth:`add_weighted_edge` are created as
dummy fragments with zero embedding and a synthetic
``model_id="weighted_graph"`` signature so that the structural
graph stays a valid container.

Thread safety:
    The class is **not thread-safe**. Both the structural graph
    and the weights dict are mutated without locks.

Complexity:
    * :meth:`add_weighted_edge` — O(1) amortized.
    * :meth:`get_edge_weight` — O(1) average.
    * :meth:`get_strong_neighbors` — O(d · t) where ``d`` is the
      out-degree and ``t`` is the number of edge types.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment_graph import FragmentGraph


class WeightedGraph:
    """Directed typed graph where edges carry float weights.

    Weights represent co-occurrence or reuse probability in
    ``[0.0, 1.0]``. The graph supports filtering neighbors by
    edge type and by minimum weight, making it suitable for
    confidence-thresholded routing decisions.

    Attributes:
        graph: The underlying structural
            :class:`~membrane.fragment_graph.FragmentGraph`. It
            stores fragments as nodes; this wrapper adds the
            typed, weighted edges.
        weights: Nested mapping ``weights[source][edge_type][target]
            = weight``. Mirrors the structural edges but stores the
            scalar weight for fast lookup.
    """

    def __init__(self) -> None:
        """Initialize an empty weighted graph."""
        self.graph = FragmentGraph()
        self.weights: dict[str, dict[str, dict[str, float]]] = {}

    def _ensure_node(self, content_hash: str) -> None:
        """Insert a placeholder fragment into the structural graph if missing.

        The dummy fragment carries a single-element zero embedding
        and a synthetic ``model_id="weighted_graph"`` signature.
        ``size=0`` and ``ttl=0.0`` reflect that the node carries
        no payload.

        Args:
            content_hash: Identifier of the node to ensure exists.
        """
        if self.graph.has_node(content_hash):
            return
        # Local imports keep this module importable without the
        # full data-model dependency graph at module load time.
        from membrane.fragment import Fragment
        from membrane.structural_signature import StructuralSignature

        dummy = Fragment(
            content_hash=content_hash,
            embedding=(0.0,),
            structural_signature=StructuralSignature(
                model_id="weighted_graph",
                layer_range=(0, 0),
                token_span=(0, 0),
            ),
            size=0,
            ttl=0.0,
            reuse_score=0.0,
            version_id=1,
        )
        self.graph.add_node(dummy)

    def add_weighted_edge(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
        weight: float,
    ) -> None:
        """Add a typed edge with a weight.

        Both endpoints are ensured to exist in the underlying
        structural graph before the edge is recorded.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Relationship type (e.g., ``"co_access"``,
                ``"prefetch"``, ``"replicates"``).
            weight: Probability or strength in ``[0.0, 1.0]``. No
                validation is performed — callers are responsible
                for keeping weights within range.
        """
        # Ensure both nodes exist in the underlying graph so that
        # add_edge never sees a missing endpoint.
        self._ensure_node(source_hash)
        self._ensure_node(target_hash)
        self.graph.add_edge(source_hash, target_hash, edge_type)
        self.weights.setdefault(source_hash, {}).setdefault(edge_type, {})[
            target_hash
        ] = weight

    def get_edge_weight(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
    ) -> float:
        """Return the weight of a typed edge.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Relationship type.

        Returns:
            float: Edge weight, or ``0.0`` if no such edge exists.
        """
        return (
            self.weights.get(source_hash, {}).get(edge_type, {}).get(target_hash, 0.0)
        )

    def get_strong_neighbors(
        self,
        content_hash: str,
        edge_type: str | None = None,
        min_weight: float = 0.5,
    ) -> set[str]:
        """Return neighbors with edge weight above a threshold.

        Args:
            content_hash: Source node hash.
            edge_type: Optional edge type filter. When ``None``,
                neighbors are collected across all edge types the
                source participates in.
            min_weight: Minimum weight threshold (inclusive).

        Returns:
            set[str]: Strong-neighbor hashes. Empty when the
            source has no edges or no edge meets the threshold.
        """
        result: set[str] = set()
        types = self.weights.get(content_hash, {})
        if edge_type is not None:
            # Restrict to a single edge type.
            for target, weight in types.get(edge_type, {}).items():
                if weight >= min_weight:
                    result.add(target)
        else:
            # Aggregate across all edge types, deduplicating
            # targets that appear with multiple types.
            for type_weights in types.values():
                for target, weight in type_weights.items():
                    if weight >= min_weight:
                        result.add(target)
        return result

    def has_node(self, content_hash: str) -> bool:
        """Check whether a node exists in the underlying structural graph.

        Args:
            content_hash: Hash of the node to check.

        Returns:
            bool: True if the node has been added (either
            explicitly or as a side effect of
            :meth:`add_weighted_edge`).
        """
        return self.graph.has_node(content_hash)
