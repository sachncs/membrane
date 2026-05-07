"""Semantic index: brute-force cosine similarity over embeddings."""

import logging

logger = logging.getLogger(__name__)


import math
from collections.abc import Sequence

from membrane.fragment import Fragment


def compute_norm(embedding: Sequence[float]) -> float:
    """Compute the L2 norm of an embedding."""
    norm = math.sqrt(sum(x * x for x in embedding))
    return norm if norm > 0.0 else 1.0


class SemanticIndex:
    """In-memory semantic index using brute-force cosine similarity.

    No external dependencies. Optional faiss integration belongs in extensions.

    .. note::
        This class is **not thread-safe**.  The internal ``fragments``
        list is not protected by locks.  If the index is accessed from
        multiple threads, the caller must provide external
        synchronisation.
    """

    def __init__(self) -> None:
        self.fragments: list[Fragment] = []
        self.norms: dict[str, float] = {}

    def insert(self, fragment: Fragment) -> None:
        """Insert a fragment into the index, caching its embedding norm."""
        self.fragments.append(fragment)
        self.norms[fragment.content_hash] = compute_norm(fragment.embedding)

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from the index.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            True if the fragment was present and removed.
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
        """Return the k fragments most similar to the query embedding.

        Args:
            query_embedding: Dense query vector.
            k: Number of neighbors to return.

        Returns:
            Fragments sorted by descending similarity.
        """
        if not self.fragments:
            return []

        query_norm = compute_norm(query_embedding)

        def cosine_similarity(frag: Fragment) -> float:
            """Compute cosine similarity using cached norms."""
            dot = sum(x * y for x, y in zip(query_embedding, frag.embedding))
            norm_b = self.norms.get(frag.content_hash, 1.0)
            return dot / (query_norm * norm_b)

        scored = [(cosine_similarity(frag), frag) for frag in self.fragments]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [frag for _, frag in scored[:k]]
