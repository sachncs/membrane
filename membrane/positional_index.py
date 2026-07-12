"""Positional index: token_span interval lookup for adjacency/overlap.

This module wraps :class:`~membrane.interval_tree.IntervalTree` to
provide a fragment-keyed positional index. It answers two kinds of
queries efficiently:

* **Overlap**: which fragments have a ``token_span`` that overlaps
  a given ``[start, end]`` range?
* **Adjacency**: which fragments are within ``max_gap`` tokens of a
  given position?

The underlying interval tree is an AVL-based structure with
max-end augmentation, giving O(log n) insertion/removal and
O(log n + m) queries where ``m`` is the number of matches.

Use cases:
    * Prefix reconstruction: find the cached fragments that cover
      the requested token range, or the nearest neighbor when no
      fragment covers the exact range.
    * Cache-friendly prefetching: discover adjacent fragments that
      could be loaded alongside a hot fragment.

Thread safety:
    The class is **not thread-safe** — operations on the interval
    tree mutate internal pointers. Provide external locking in
    concurrent environments.
"""

import logging

logger = logging.getLogger(__name__)

from membrane.fragment import Fragment
from membrane.interval_tree import IntervalTree


class PositionalIndex:
    """In-memory positional index using an interval tree for overlap queries.

    Provides O(log n + m) overlap and adjacency lookups via an
    AVL-based interval tree with max-end augmentation.

    .. note::
        This class is **not thread-safe**.  If the index is
        accessed from multiple threads, the caller must provide
        external synchronisation.
    """

    def __init__(self) -> None:
        """Initialize an empty positional index backed by an interval tree."""
        self.tree = IntervalTree()

    def insert(self, fragment: Fragment) -> None:
        """Insert a fragment into the index.

        The fragment's ``structural_signature.token_span`` is used
        as the interval key. Inserting a fragment whose hash is
        already present overwrites the previous entry.

        Args:
            fragment: Fragment to insert.
        """
        self.tree.insert(fragment)

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from the index.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            bool: True if the fragment was present and removed,
            False otherwise.
        """
        return self.tree.remove(content_hash)

    def find_overlapping(
        self,
        start: int,
        end: int,
    ) -> list[Fragment]:
        """Find fragments whose ``token_span`` overlaps ``[start, end]``.

        Uses the interval tree for O(log n + m) performance.

        Args:
            start: Query start position (inclusive).
            end: Query end position (inclusive).

        Returns:
            list[Fragment]: Fragments with overlapping token
            spans. Order is unspecified.
        """
        return self.tree.find_overlapping(start, end)

    def find_adjacent(
        self,
        position: int,
        max_gap: int = 0,
    ) -> list[Fragment]:
        """Find fragments adjacent to a given position.

        A fragment is *adjacent* if its ``token_span`` starts within
        ``max_gap`` tokens after ``position``, or ends within
        ``max_gap`` tokens before it. With ``max_gap = 0`` only
        touching fragments are returned.

        Args:
            position: Reference token position.
            max_gap: Maximum allowed gap for adjacency.

        Returns:
            list[Fragment]: Adjacent fragments in interval-tree
            traversal order.
        """
        return self.tree.find_adjacent(position, max_gap)
