"""SemanticCluster: group similar prefixes by embedding similarity.

This module implements a simple greedy clustering algorithm that
groups fragments whose embeddings are similar above a configurable
threshold. The clustering is intended for low-cardinality workloads
(``k`` in the hundreds to low thousands); for larger collections
swap in a proper clustering algorithm (DBSCAN, HDBSCAN, k-means)
or rely on
:class:`~membrane.semantic_index.SemanticIndex`'s top-K search.

Algorithm:
    1. Insert every fragment into the supplied
       :class:`~membrane.semantic_index.SemanticIndex` for fast
       neighbor lookup.
    2. Repeatedly pick the lowest-index unassigned fragment as the
       *seed* of a new cluster.
    3. Greedily assign every other unassigned fragment whose
       cosine similarity to the seed exceeds
       ``similarity_threshold`` to the cluster.
    4. Continue until every fragment is assigned.

Complexity:
    * Outer loop runs ``len(fragments)`` times.
    * Each iteration performs a semantic lookup per unassigned
      fragment. Overall complexity is O(n^2 · d) for ``n``
      fragments of dimensionality ``d`` — acceptable for the
      intended workload sizes.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.semantic_index import SemanticIndex


class SemanticCluster:
    """Groups fragments into clusters based on embedding similarity.

    The class is intentionally stateless beyond its
    :class:`~membrane.semantic_index.SemanticIndex` reference, so
    instances can be shared across threads as long as the supplied
    index is itself thread-safe (it is not).
    """

    def __init__(self, semantic_index: SemanticIndex | None = None) -> None:
        """Initialize the clusterer with an optional semantic index.

        Args:
            semantic_index: Index for similarity lookups. A
                default empty
                :class:`~membrane.semantic_index.SemanticIndex` is
                created when ``None``.
        """
        self.semantic_index = semantic_index or SemanticIndex()

    def cluster(
        self,
        fragments: list[Fragment],
        similarity_threshold: float = 0.95,
    ) -> list[list[Fragment]]:
        """Group ``fragments`` into clusters of similar embeddings.

        The first fragment becomes the seed of the first cluster;
        the algorithm then grows the cluster greedily, adding any
        unassigned fragment whose cosine similarity to the seed is
        at least ``similarity_threshold``. Once no more fragments
        can be added, the next unassigned fragment seeds a new
        cluster, and so on.

        Args:
            fragments: Fragments to cluster. May be empty.
            similarity_threshold: Minimum cosine similarity within
                a cluster, in ``[0, 1]``. Values close to ``1``
                yield tight clusters; lower values produce broader
                groupings.

        Returns:
            list[list[Fragment]]: List of fragment clusters. Empty
            when ``fragments`` is empty.
        """
        if not fragments:
            return []

        # Build the semantic index once so we can use its cached
        # norms during the greedy expansion.
        for frag in fragments:
            self.semantic_index.insert(frag)

        unassigned = set(range(len(fragments)))
        clusters: list[list[Fragment]] = []

        while unassigned:
            # Pick the lowest-index unassigned fragment as the
            # deterministic seed for this cluster.
            seed_idx = min(unassigned)
            seed = fragments[seed_idx]
            cluster = [seed]
            unassigned.remove(seed_idx)

            # Greedy expansion: scan unassigned candidates,
            # comparing each to the seed.
            to_remove: list[int] = []
            for idx in list(unassigned):
                candidate = fragments[idx]
                # First, use the semantic index to short-circuit
                # the trivial case: the seed is its own nearest
                # neighbor, so a top-1 match on the seed means
                # the candidate is at least as close as the seed
                # is to itself.
                neighbors = self.semantic_index.nearest_neighbors(list(candidate.embedding), k=1)
                if neighbors and neighbors[0].content_hash == seed.content_hash:
                    cluster.append(candidate)
                    to_remove.append(idx)
                else:
                    # Otherwise fall back to a direct cosine
                    # comparison against the threshold.
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
            b: Second embedding. Must have the same dimensionality
                as ``a``; extra elements in the longer vector are
                ignored by :func:`zip`.

        Returns:
            float: Cosine similarity in ``[-1.0, 1.0]``. Returns
            ``0.0`` when either vector has zero norm.
        """
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            # Avoid division by zero for zero-norm vectors; treat
            # them as orthogonal to everything.
            return 0.0
        return dot / (norm_a * norm_b)
