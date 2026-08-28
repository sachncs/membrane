"""Semantic index: brute-force cosine similarity over embeddings.

This module implements :class:`SemanticIndex`, a small in-memory
nearest-neighbor index over fragment embeddings. It uses brute-force
cosine similarity — every query scores against every indexed
fragment — which is appropriate when the number of fragments is
modest (thousands to tens of thousands). For larger collections,
swap in a proper ANN index (e.g., FAISS) by implementing the
:class:`~membrane.protocols.IndexProtocol` interface.

The index caches each fragment's embedding L2 norm so that the
similarity computation only needs to multiply the query embedding
by each fragment embedding, rather than recomputing norms.

Thread safety:
    Like :class:`~membrane.exact_index.ExactIndex`, this class is
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


import math
from collections.abc import Sequence

from membrane.fragment import Fragment


def compute_norm(embedding: Sequence[float]) -> float:
    """Compute the L2 norm of an embedding, treating zero as one.

    A zero norm is replaced by ``1.0`` so that downstream cosine
    similarity divisions never divide by zero. In practice, all
    fragments produced by
    :func:`membrane.fragmentation_engine.generate_embedding` are
    unit-normalized, so this branch is never taken in the normal
    pipeline.

    Args:
        embedding: Dense vector.

    Returns:
        float: L2 norm of the embedding, or ``1.0`` if the norm is
        zero.
    """
    norm = math.sqrt(sum(x * x for x in embedding))
    return norm if norm > 0.0 else 1.0


class SemanticIndex:
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
        """Insert a fragment into the index, caching its embedding norm.

        The fragment is appended to the internal list and its
        embedding norm is computed once and cached in ``self.norms``
        so that :meth:`nearest_neighbors` can avoid recomputing it
        on every query.

        Args:
            fragment: The fragment to index.
        """
        self.fragments.append(fragment)
        self.norms[fragment.content_hash] = compute_norm(fragment.embedding)

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
            if frag.content_hash == content_hash:
                self.fragments.pop(i)
                self.norms.pop(content_hash, None)
                return True
        return False

    def nearest_neighbors(
        self,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> list[Fragment]:
        """Return the k fragments most similar to ``query_embedding``.

        Cosine similarity is used as the metric. The function
        sorts all fragments by descending similarity and returns
        the top ``k``.

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

        query_norm = compute_norm(query_embedding)

        def cosine_similarity(frag: Fragment) -> float:
            """Cosine similarity between query and a single fragment.

            Uses the cached fragment norm to avoid recomputing it
            per query.
            """
            dot = sum(x * y for x, y in zip(query_embedding, frag.embedding, strict=False))
            norm_b = self.norms.get(frag.content_hash, 1.0)
            return dot / (query_norm * norm_b)

        # Score all fragments; sort descending by similarity.
        scored = [(cosine_similarity(frag), frag) for frag in self.fragments]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [frag for _, frag in scored[:k]]
