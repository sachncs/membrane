"""LRU cache: bounded least-recently-used key tracker.

Provides touch tracking and overflow eviction.  Composed by stores that
need bounded LRU behavior.
"""

import time


class LRUCache:
    """Tracks access times for keys and evicts the oldest when over capacity.

    This class does *not* store the values themselves; it only tracks
    access ordering and tells the caller which key to evict.
    """

    def __init__(self, capacity: int | None = None) -> None:
        """Initialize the LRU tracker.

        Args:
            capacity: Maximum number of keys to track.  None means unbounded.
        """
        self.capacity = capacity
        self.access_times: dict[str, float] = {}

    def touch(self, key: str) -> None:
        """Record that *key* was just accessed."""
        self.access_times[key] = time.time()

    def evict_if_over(self) -> list[str]:
        """Evict oldest keys while count exceeds capacity.

        Returns:
            List of evicted keys.
        """
        if self.capacity is None:
            return []
        evicted: list[str] = []
        while len(self.access_times) > self.capacity:
            if not self.access_times:
                break
            oldest = min(self.access_times, key=lambda k: self.access_times[k])
            self.access_times.pop(oldest, None)
            evicted.append(oldest)
        return evicted

    def remove(self, key: str) -> None:
        """Remove a key from tracking."""
        self.access_times.pop(key, None)
