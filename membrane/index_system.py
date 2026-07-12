"""IndexSystem: facade over all four in-memory indices.

This module defines :class:`IndexSystem`, a thin aggregate that
coordinates the four specialized index structures that ship with
Membrane:

* :class:`~membrane.exact_index.ExactIndex` — primary
  ``content_hash`` lookup with replica tracking.
* :class:`~membrane.semantic_index.SemanticIndex` — embedding-based
  similarity search.
* :class:`~membrane.positional_index.PositionalIndex` — token-span
  overlap and adjacency queries.
* :class:`~membrane.co_access_index.CoAccessIndex` — undirected
  graph of fragments accessed together.

The facade ensures that *every* sub-index stays in sync whenever a
fragment is inserted or removed, and exposes a single set of
query methods that map to the most appropriate sub-index.

Because the underlying indexes are independent data structures,
:class:`IndexSystem` can be initialized lazily in tests or replaced
with mocks that satisfy
:class:`~membrane.protocols.IndexProtocol`.

Thread safety:
    The class inherits the threading properties of its
    sub-indexes. None of them are individually thread-safe, so the
    facade as a whole is not thread-safe.
"""

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

    Provides incremental updates and supports distributed queries
    by exposing each sub-index independently.

    Attributes:
        exact: Primary ``content_hash`` -> entry index.
        semantic: Embedding-based similarity index.
        positional: Token-span overlap/adjacency index.
        co_access: Undirected co-access graph.
    """

    def __init__(self) -> None:
        """Initialize empty sub-indices and log the lifecycle event."""
        logger.info("Initialized %s", self.__class__.__name__)
        self.exact = ExactIndex()
        self.semantic = SemanticIndex()
        self.positional = PositionalIndex()
        self.co_access = CoAccessIndex()

    def insert(self, fragment: Fragment, locations: set[str]) -> None:
        """Insert a fragment into all indices.

        The exact index receives both the fragment and its
        locations; the semantic and positional indexes only need
        the fragment itself. The co-access index is updated
        separately via :meth:`record_co_access` so callers can
        distinguish structural insertion from behavioral
        observation.

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
            IndexEntry | None: The matching entry, or ``None`` if
            no fragment with that hash is indexed.
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
            list[Fragment]: Fragments sorted by descending
            similarity. Empty when the index has no entries.
        """
        return self.semantic.nearest_neighbors(query_embedding, k)

    def positional_lookup(self, start: int, end: int) -> list[Fragment]:
        """Find fragments overlapping a token span.

        Args:
            start: Query start position (inclusive).
            end: Query end position (inclusive).

        Returns:
            list[Fragment]: Fragments with overlapping token
            spans. Order is unspecified.
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
            list[Fragment]: Adjacent fragments in traversal order.
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

        Each sub-index is queried independently because their
        membership sets may diverge (e.g., a co-access entry can
        outlive its fragment). The method returns ``True`` if at
        least one sub-index reported a removal.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            bool: True if the fragment was found in at least one
            sub-index, False otherwise.
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
            set[str]: Defensive copy of the neighbor set. Empty
            when the hash has no recorded co-accesses.
        """
        return self.co_access.lookup(content_hash)
