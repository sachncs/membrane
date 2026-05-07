"""IndexSystem: facade over all four in-memory indices."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


from collections.abc import Sequence

from membrane.co_access_index import CoAccessIndex
from membrane.exact_index import ExactIndex, IndexEntry
from membrane.fragment import Fragment
from membrane.positional_index import PositionalIndex
from membrane.semantic_index import SemanticIndex


class IndexSystem:
    """Manages exact, semantic, positional, and co-access indices.

    Provides incremental updates and supports distributed queries by
    exposing each sub-index independently.
    """

    def __init__(self) -> None:
        logger.info("Initialized %s", self.__class__.__name__)
        self.exact = ExactIndex()
        self.semantic = SemanticIndex()
        self.positional = PositionalIndex()
        self.co_access = CoAccessIndex()

    def insert(self, fragment: Fragment, locations: set[str]) -> None:
        """Insert a fragment into all indices.

        Args:
            fragment: Fragment to index.
            locations: Set of node IDs holding the fragment.
        """
        self.exact.insert(fragment, locations)
        self.semantic.insert(fragment)
        self.positional.insert(fragment)

    def exact_lookup(self, content_hash: str) -> IndexEntry | None:
        """Look up a fragment by exact content hash.

        Args:
            content_hash: Hash to look up.

        Returns:
            IndexEntry if found, else None.
        """
        return self.exact.lookup(content_hash)

    def semantic_lookup(
        self,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> list[Fragment]:
        """Find fragments by semantic similarity.

        Args:
            query_embedding: Dense query vector.
            k: Number of neighbors to return.

        Returns:
            List of fragments sorted by descending similarity.
        """
        return self.semantic.nearest_neighbors(query_embedding, k)

    def positional_lookup(self, start: int, end: int) -> list[Fragment]:
        """Find fragments overlapping a token span.

        Args:
            start: Query start position (inclusive).
            end: Query end position (inclusive).

        Returns:
            Fragments with overlapping token spans.
        """
        return self.positional.find_overlapping(start, end)

    def positional_adjacent(
        self,
        position: int,
        max_gap: int = 0,
    ) -> list[Fragment]:
        """Find fragments adjacent to a token position.

        Args:
            position: Reference token position.
            max_gap: Maximum allowed gap for adjacency.

        Returns:
            Adjacent fragments.
        """
        return self.positional.find_adjacent(position, max_gap)

    def record_co_access(self, hash_a: str, hash_b: str) -> None:
        """Record that two fragments were accessed together.

        Args:
            hash_a: Content hash of the first fragment.
            hash_b: Content hash of the second fragment.
        """
        self.co_access.record_access(hash_a, hash_b)

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from all sub-indices.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            True if the fragment was found in at least one index.
        """
        removed = (
            self.exact.remove(content_hash)
            or self.semantic.remove(content_hash)
            or self.positional.remove(content_hash)
            or self.co_access.remove(content_hash)
        )
        return removed

    def co_access_neighbors(self, content_hash: str) -> set[str]:
        """Return co-access neighbors of a fragment.

        Args:
            content_hash: Hash to look up.

        Returns:
            Set of neighbor hashes.
        """
        return self.co_access.lookup(content_hash)
