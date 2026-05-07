"""SemanticCluster: group similar prefixes by embedding similarity."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.semantic_index import SemanticIndex


class SemanticCluster:
    """Groups fragments into clusters based on embedding similarity."""

    def __init__(self, semantic_index: SemanticIndex | None = None) -> None:
        """Initialize with optional semantic index.

        Args:
            semantic_index: Index for similarity lookups.
        """
        """Initialize with optional semantic index.

        Args:
            semantic_index: Index for similarity lookups.
        """
        self.semantic_index = semantic_index or SemanticIndex()

    def cluster(
        self,
        fragments: list[Fragment],
        similarity_threshold: float = 0.95,
    ) -> list[list[Fragment]]:
        """Group fragments into clusters where each member is similar to the centroid.

        Args:
            fragments: Fragments to cluster.
            similarity_threshold: Minimum cosine similarity within a cluster.

        Returns:
            List of fragment clusters.
        """
        if not fragments:
            return []

        # Build index for fast neighbor lookup
        for frag in fragments:
            self.semantic_index.insert(frag)

        unassigned = set(range(len(fragments)))
        clusters: list[list[Fragment]] = []

        while unassigned:
            seed_idx = min(unassigned)
            seed = fragments[seed_idx]
            cluster = [seed]
            unassigned.remove(seed_idx)

            # Greedy expansion: add all unassigned fragments similar to seed
            to_remove: list[int] = []
            for idx in list(unassigned):
                candidate = fragments[idx]
                # Use semantic index for approximate similarity
                neighbors = self.semantic_index.nearest_neighbors(
                    list(candidate.embedding), k=1
                )
                if neighbors and neighbors[0].content_hash == seed.content_hash:
                    # Self-match means it's very similar to itself; use exact comparison
                    cluster.append(candidate)
                    to_remove.append(idx)
                else:
                    # Simple cosine comparison for threshold
                    sim = self.cosine_similarity(seed.embedding, candidate.embedding)
                    if sim >= similarity_threshold:
                        cluster.append(candidate)
                        to_remove.append(idx)

            for idx in to_remove:
                unassigned.discard(idx)

            clusters.append(cluster)

        return clusters

    @staticmethod
    def cosine_similarity(
        a: tuple[float, ...],
        b: tuple[float, ...],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            a: First embedding.
            b: Second embedding.

        Returns:
            Cosine similarity in [-1.0, 1.0].
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
