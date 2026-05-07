"""WeightedGraph: graph edges with co-occurrence probabilities."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment_graph import FragmentGraph


class WeightedGraph:
    """Directed typed graph where edges carry float weights.

    Weights represent co-occurrence or reuse probability.
    """

    def __init__(self) -> None:
        """Initialize an empty weighted graph."""
        """Initialize an empty weighted graph."""
        self.graph = FragmentGraph()
        self.weights: dict[str, dict[str, dict[str, float]]] = {}

    def add_weighted_edge(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
        weight: float,
    ) -> None:
        """Add a typed edge with a weight.

        Args:
            source_hash: Source fragment hash.
            target_hash: Target fragment hash.
            edge_type: Relationship type.
            weight: Probability or strength in [0.0, 1.0].
        """
        # Ensure both nodes exist in the underlying graph
        if not self.graph.has_node(source_hash):
            from membrane.fragment import Fragment
            from membrane.structural_signature import StructuralSignature

            dummy = Fragment(
                content_hash=source_hash,
                embedding=(0.0,),
                structural_signature=StructuralSignature(
                    model_id="weighted_graph", layer_range=(0, 0), token_span=(0, 0)
                ),
                size=0,
                ttl=0.0,
                reuse_score=0.0,
                version_id=1,
            )
            self.graph.add_node(dummy)
        if not self.graph.has_node(target_hash):
            from membrane.fragment import Fragment
            from membrane.structural_signature import StructuralSignature

            dummy = Fragment(
                content_hash=target_hash,
                embedding=(0.0,),
                structural_signature=StructuralSignature(
                    model_id="weighted_graph", layer_range=(0, 0), token_span=(0, 0)
                ),
                size=0,
                ttl=0.0,
                reuse_score=0.0,
                version_id=1,
            )
            self.graph.add_node(dummy)
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
            Edge weight, or 0.0 if no edge exists.
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
        """Return neighbors with edge weight above threshold.

        Args:
            content_hash: Source node hash.
            edge_type: Optional edge type filter.
            min_weight: Minimum weight threshold.

        Returns:
            Set of strong neighbor hashes.
        """
        result: set[str] = set()
        types = self.weights.get(content_hash, {})
        if edge_type is not None:
            for target, weight in types.get(edge_type, {}).items():
                if weight >= min_weight:
                    result.add(target)
        else:
            for type_weights in types.values():
                for target, weight in type_weights.items():
                    if weight >= min_weight:
                        result.add(target)
        return result

    def has_node(self, content_hash: str) -> bool:
        """Check if a node exists in the graph."""
        return self.graph.has_node(content_hash)
