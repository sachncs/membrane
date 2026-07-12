"""LRU cache: bounded least-recently-used key tracker.

Provides touch tracking and overflow eviction.  Composed by stores
that need bounded LRU behavior.

Unlike a typical :class:`collections.OrderedDict`-based LRU, this
class is a *tracker* — it does not store values. It records the
last-access time for each key and tells the caller which key to
evict when the configured capacity is exceeded. The caller (e.g.,
:class:`membrane.fragment_store.FragmentStore`) is responsible for
deleting the corresponding payload from its own storage.

Thread safety:
    The class is **not thread-safe**. Provide external locking
    when sharing across threads.
"""

import time


class LRUCache:
    """Tracks access times for keys and evicts the oldest when over capacity.

    This class does *not* store the values themselves; it only
    tracks access ordering and tells the caller which key to
    evict.

    Attributes:
        capacity: Maximum number of keys to track. ``None`` means
            unbounded (no eviction).
        access_times: Mapping from key to the timestamp of its
            most recent access (as returned by :func:`time.time`).
    """

    def __init__(self, capacity: int | None = None) -> None:
        """Initialize the LRU tracker.

        Args:
            capacity: Maximum number of keys to track. ``None``
                means unbounded — :meth:`evict_if_over` becomes a
                no-op.
        """
        self.capacity = capacity
        self.access_times: dict[str, float] = {}

    def touch(self, key: str) -> None:
        """Record that ``key`` was just accessed.

        Args:
            key: Identifier of the touched entry.
        """
        self.access_times[key] = time.time()

    def evict_if_over(self) -> list[str]:
        """Evict oldest keys while count exceeds capacity.

        Repeatedly finds and removes the key with the smallest
        ``access_time`` until the count drops to ``capacity`` or
        the cache is empty.

        Returns:
            list[str]: Keys that were evicted, in eviction order.
            Empty when the cache is at or under capacity, or when
            ``capacity`` is ``None``.
        """
        if self.capacity is None:
            return []
        evicted: list[str] = []
        while len(self.access_times) > self.capacity:
            if not self.access_times:
                break
            # min() with a key= callback gives O(n) per removal;
            # the entire call is O((n - capacity) * n) in the
            # worst case, which is acceptable for the modest
            # capacities typical of LRU trackers.
            oldest = min(self.access_times, key=lambda k: self.access_times[k])
            self.access_times.pop(oldest, None)
            evicted.append(oldest)
        return evicted

    def remove(self, key: str) -> None:
        """Remove a key from tracking.

        Args:
            key: Identifier to remove. No-op if the key is not
                tracked.
        """
        self.access_times.pop(key, None)
