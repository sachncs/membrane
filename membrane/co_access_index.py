"""Co-access index: graph backbone tracking fragments accessed together."""

import logging

logger = logging.getLogger(__name__)


class CoAccessIndex:
    """In-memory co-access index using adjacency sets.

    Records which content hashes are accessed together and supports
    neighbor lookups for prefetch and replication grouping.

    .. note::
        This class is **not thread-safe**.  The internal ``graph`` dict
        is not protected by locks.  If the index is accessed from
        multiple threads, the caller must provide external
        synchronisation.
    """

    def __init__(self) -> None:
        self.graph: dict[str, set[str]] = {}

    def record_access(self, hash_a: str, hash_b: str) -> None:
        """Record that two fragments were accessed together.

        Args:
            hash_a: Content hash of the first fragment.
            hash_b: Content hash of the second fragment.
        """
        if hash_a == hash_b:
            return
        self.graph.setdefault(hash_a, set()).add(hash_b)
        self.graph.setdefault(hash_b, set()).add(hash_a)

    def lookup(self, content_hash: str) -> set[str]:
        """Return all content hashes co-accessed with the given one.

        Args:
            content_hash: Hash to look up.

        Returns:
            Set of neighbor hashes.
        """
        return set(self.graph.get(content_hash, set()))

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment and all its co-access edges.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            True if the fragment had any recorded co-access relationships.
        """
        neighbors = self.graph.pop(content_hash, set())
        for neighbor in neighbors:
            self.graph.get(neighbor, set()).discard(content_hash)
        return len(neighbors) > 0

    def record_batch(self, hashes: list[str]) -> None:
        """Record co-access for all pairs in a batch.

        Args:
            hashes: List of content hashes accessed in the same request.
        """
        unique = list(dict.fromkeys(hashes))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                self.record_access(unique[i], unique[j])
