"""Positional index: token_span interval lookup for adjacency/overlap.

Delegates to ``IntervalTree`` for O(log n + m) overlap and adjacency
queries, plus O(log n) insertion and deletion.
"""

import logging

logger = logging.getLogger(__name__)

from membrane.fragment import Fragment
from membrane.interval_tree import IntervalTree


class PositionalIndex:
    """In-memory positional index using an interval tree for overlap queries.

    Provides O(log n + m) overlap and adjacency lookups via an AVL-based
    interval tree with max-end augmentation.

    .. note::
        This class is **not thread-safe**.  If the index is accessed from
        multiple threads, the caller must provide external synchronisation.
    """

    def __init__(self) -> None:
        self.tree = IntervalTree()

    def insert(self, fragment: Fragment) -> None:
        """Insert a fragment into the index."""
        self.tree.insert(fragment)

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from the index.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            True if the fragment was present and removed.
        """
        return self.tree.remove(content_hash)

    def find_overlapping(
        self,
        start: int,
        end: int,
    ) -> list[Fragment]:
        """Find fragments whose token_span overlaps [start, end].

        Uses the interval tree for O(log n + m) performance.

        Args:
            start: Query start position (inclusive).
            end: Query end position (inclusive).

        Returns:
            Fragments with overlapping token spans.
        """
        return self.tree.find_overlapping(start, end)

    def find_adjacent(
        self,
        position: int,
        max_gap: int = 0,
    ) -> list[Fragment]:
        """Find fragments adjacent to a given position.

        A fragment is adjacent if its token_span starts within max_gap tokens
        after the given position, or ends within max_gap tokens before it.

        Args:
            position: Reference token position.
            max_gap: Maximum allowed gap for adjacency.

        Returns:
            Adjacent fragments.
        """
        return self.tree.find_adjacent(position, max_gap)
