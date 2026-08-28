"""Co-access index: graph backbone tracking fragments accessed together.

This module implements :class:`CoAccessIndex`, an undirected graph
structure that records which fragments are accessed together in the
same request. The resulting co-access relationships power two
optimizations in Membrane:

* **Prefetching**: when fragment ``a`` is requested, fragments in
  its co-access neighborhood are likely to be requested next, so
  they can be proactively fetched.
* **Replication grouping**: co-accessed fragments benefit from being
  placed on the same node to minimize cross-node traffic.

Internally the graph is represented as an adjacency-set mapping:
``graph[hash]`` is the set of hashes that have been co-accessed with
``hash``. The implementation keeps the two edges ``(a, b)`` and
``(b, a)`` in sync so callers can query from either endpoint.

Thread safety:
    The class is **not thread-safe**.  The internal ``graph`` dict
    is not protected by locks.  If the index is accessed from
    multiple threads, the caller must provide external
    synchronisation.

Complexity:
    * :meth:`record_access` — O(1) amortized per edge.
    * :meth:`lookup` — O(1) plus copy of the neighbor set.
    * :meth:`record_batch` — O(k^2) where ``k`` is the number of
      distinct hashes in the batch.
"""

import logging

logger = logging.getLogger(__name__)


class CoAccessIndex:
    """In-memory co-access index using adjacency sets.

    Records which content hashes are accessed together and supports
    neighbor lookups for prefetch and replication grouping.

    .. note::
        This class is **not thread-safe**.  The internal ``graph``
        dict is not protected by locks.  If the index is accessed
        from multiple threads, the caller must provide external
        synchronisation.

    Attributes:
        graph: Adjacency-set representation. ``graph[a]`` contains
            all hashes ``b`` such that ``record_access(a, b)`` (or
            ``record_access(b, a)``) has been called.
    """

    def __init__(self) -> None:
        """Initialize an empty co-access graph."""
        self.graph: dict[str, set[str]] = {}

    def record_access(self, hash_a: str, hash_b: str) -> None:
        """Record that two fragments were accessed together.

        Self-loops (``hash_a == hash_b``) are ignored — a fragment
        is not considered a co-access neighbor of itself.

        Args:
            hash_a: Content hash of the first fragment.
            hash_b: Content hash of the second fragment.
        """
        if hash_a == hash_b:
            return
        # Record the edge in both directions so either endpoint
        # can be used as a query key.
        self.graph.setdefault(hash_a, set()).add(hash_b)
        self.graph.setdefault(hash_b, set()).add(hash_a)

    def lookup(self, content_hash: str) -> set[str]:
        """Return all content hashes co-accessed with ``content_hash``.

        Args:
            content_hash: Hash to look up.

        Returns:
            set[str]: A defensive copy of the neighbor set. Empty
            when the hash has no recorded co-accesses.
        """
        return set(self.graph.get(content_hash, set()))

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment and all its co-access edges.

        Iterates over the fragment's neighbors and removes the
        reverse edge from each one.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            bool: True if the fragment had any recorded
            co-access relationships, False otherwise.
        """
        neighbors = self.graph.pop(content_hash, set())
        for neighbor in neighbors:
            self.graph.get(neighbor, set()).discard(content_hash)
        return len(neighbors) > 0

    def record_batch(self, hashes: list[str]) -> None:
        """Record co-access for all pairs in a batch.

        Duplicates in ``hashes`` are removed (preserving order)
        before pairing, so a fragment is never paired with itself
        and pairs are only recorded once.

        Args:
            hashes: List of content hashes accessed in the same
                request. Order is not significant.
        """
        # dict.fromkeys preserves insertion order while removing
        # duplicates in a single pass.
        unique = list(dict.fromkeys(hashes))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                self.record_access(unique[i], unique[j])
