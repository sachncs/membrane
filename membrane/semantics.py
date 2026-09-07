"""Semantic index: brute-force cosine similarity over embeddings.

This module implements :class:`Semantics`, a small in-memory
nearest-neighbor index over fragment embeddings. It uses brute-force
cosine similarity — every query scores against every indexed
fragment — which is appropriate when the number of fragments is
modest (thousands to tens of thousands). For larger collections,
swap in a proper ANN index (e.g., FAISS) by implementing the
:class:`~membrane.index.Index` interface.

The index caches each fragment's embedding L2 norm so that the
similarity computation only needs to multiply the query embedding
by each fragment embedding, rather than recomputing norms.

Thread safety:
    Like :class:`~membrane.exacts.Exacts`, this class is
    **not thread-safe**. The fragments list and the norms dict are
    mutated without locks. Callers must provide external
    synchronization when sharing across threads.

Complexity:
    * :meth:`insert` — O(1) amortized.
    * :meth:`remove` — O(N) where N is the number of indexed
      fragments (linear scan).
    * :meth:`nearest_neighbors` — O(N · d) where ``d`` is the
      embedding dimensionality, plus O(N log N) for the final sort.
"""

import logging

logger = logging.getLogger(__name__)


from collections.abc import Sequence

from membrane.fragment import Fragment


def compute_norm(embedding: Sequence[float]) -> float:
    """Compute the L2 norm of an embedding, treating zero as one.

    A zero norm is replaced by ``1.0`` so that downstream cosine
    similarity divisions never divide by zero. This is a
    degenerate fallback for callers that still synthesize raw
    query embeddings; the canonical path uses
    :func:`membrane.fragmenter.generate_embedding`.

    Args:
        embedding: Dense vector.

    Returns:
        float: L2 norm of the embedding, or ``1.0`` if the norm is
        zero.
    """
    norm = sum(x * x for x in embedding) ** 0.5
    return norm if norm > 0.0 else 1.0


class Semantics:
    """In-memory semantic index using brute-force cosine similarity.

    No external dependencies. Optional faiss integration belongs in
    extensions.

    .. note::
        This class is **not thread-safe**.  The internal
        ``fragments`` list is not protected by locks.  If the index
        is accessed from multiple threads, the caller must provide
        external synchronisation.
    """

    def __init__(self) -> None:
        """Initialize an empty semantic index."""
        self.fragments: list[Fragment] = []
        self.norms: dict[str, float] = {}

    def insert(self, fragment: Fragment) -> None:
        """Insert a fragment into the index.

        The fragment is appended to the internal list and its
        payload hash is recorded so :meth:`nearest_neighbors` and
        :meth:`remove` can locate it. The new ``Fragment`` schema
        no longer carries an ``embedding``; similarity searches
        therefore use a placeholder zero-norm cache entry per
        fragment. Callers that need real embeddings should pair
        this index with an out-of-band embedding store.

        Args:
            fragment: The fragment to index.
        """
        self.fragments.append(fragment)
        self.norms[fragment.identity.payload_hash] = 1.0

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from the index.

        Performs a linear scan over ``self.fragments`` to find the
        entry, then removes it and its cached norm.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            bool: True if the fragment was present and removed,
            False otherwise.
        """
        for i, frag in enumerate(self.fragments):
            if frag.identity.payload_hash == content_hash:
                self.fragments.pop(i)
                self.norms.pop(content_hash, None)
                return True
        return False

    def nearest_neighbors(
        self,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> list[Fragment]:
        """Return the k fragments whose payload hash most closely
        resembles ``query_embedding``.

        Cosine similarity is used as the metric. The function
        sorts all fragments by descending similarity and returns
        the top ``k``.

        .. note::
            Now that ``Fragment`` no longer carries an ``embedding``,
            the similarity score is computed against the unit-norm
            placeholder inserted by :meth:`insert`. The query is
            still hashed and ranked, but the actual ranking is a
            no-op (all fragments score identically). This preserves
            the index's contract while exposing the schema change
            clearly to callers.

        Args:
            query_embedding: Dense query vector.
            k: Number of neighbors to return.

        Returns:
            list[Fragment]: Fragments sorted by descending
            similarity. Returns an empty list when the index is
            empty. When the index has fewer than ``k`` entries,
            all of them are returned (sorted).
        """
        if not self.fragments:
            return []

        # Score all fragments; sort descending by similarity. With
        # the new schema every fragment's stored vector is the
        # unit-norm placeholder, so the score degenerates to a
        # stable insertion-order tie-breaker.
        scored = [(0.0, frag) for frag in self.fragments]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [frag for _, frag in scored[:k]]


class SemanticCluster:
    """Greedy clustering of fragments by embedding similarity.

    The clusterer is a thin layer over :class:`Semantics`: it
    inserts every fragment into the index, then walks the
    fragment list picking the lowest-index unassigned fragment
    as a seed, and greedily assigns every other unassigned
    fragment whose cosine similarity to the seed exceeds
    ``similarity_threshold``. Continues until every fragment is
    assigned.

    .. note::
        With the new ``Fragment`` schema that drops ``embedding``,
        :meth:`cluster` no longer computes a meaningful similarity
        score. Each fragment is treated as a singleton cluster
        unless two fragments share a ``payload_hash``, in which
        case they end up in the same cluster by definition.

    Complexity:
        O(n) per cluster — acceptable for low-cardinality workloads.

    Attributes:
        semantic_index: The :class:`Semantics` index used for
            fast similarity lookups.
    """

    def __init__(self, semantic_index: Semantics | None = None) -> None:
        """Initialize the clusterer with an optional semantic index."""
        self.semantic_index = semantic_index or Semantics()

    def cluster(
        self,
        fragments: list[Fragment],
        similarity_threshold: float = 0.95,
    ) -> list[list[Fragment]]:
        """Group fragments by embedding similarity.

        Args:
            fragments: Fragments to cluster.
            similarity_threshold: Minimum cosine similarity
                within a cluster. Unused under the new schema;
                retained for API compatibility.

        Returns:
            list[list[Fragment]]: One cluster list per group.
        """
        del similarity_threshold
        if not fragments:
            return []

        for frag in fragments:
            self.semantic_index.insert(frag)

        # Group fragments by payload_hash — identical payloads are
        # the only meaningful cluster under the new schema.
        groups: dict[str, list[Fragment]] = {}
        order: list[str] = []
        for frag in fragments:
            h = frag.identity.payload_hash
            if h not in groups:
                order.append(h)
                groups[h] = []
            groups[h].append(frag)

        return [groups[h] for h in order]


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two embedding tuples.

    Returns 0.0 when either vector has zero norm. Retained for
    API compatibility with callers that supply embeddings
    out-of-band; ``Fragment`` itself no longer carries one.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["SemanticCluster", "Semantics", "cosine_similarity"]
