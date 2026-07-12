"""CacheMetrics: hit rate, size growth, and performance tracking.

This module defines :class:`CacheMetrics`, an *immutable* metrics
record for a single-region memory cache. Every update returns a
new :class:`CacheMetrics` instance rather than mutating the
existing one — a deliberate design choice so the metrics can be
safely shared across threads and so snapshots can be retained for
historical analysis.

The metrics track:

* **Hit/miss counts** — direct observations from
  :class:`~membrane.kv_cache_manager.KVCacheManager` and similar.
* **Total requests** — convenience aggregate, equal to
  ``hits + misses``.
* **Total KV bytes missed** — cumulative size of KV payloads
  that the cache failed to serve. Useful for sizing pressure.
* **Peak memory bytes** — running maximum of the cumulative
  missed-bytes counter; captures the worst-case cache footprint
  over time.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheMetrics:
    """Metrics for a single-region memory cache.

    Immutable; every update returns a new instance.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        total_requests: Total number of lookups
            (``hits + misses``).
        total_kv_size_bytes: Total size of missed KV in bytes.
        peak_memory_bytes: Peak cumulative missed size seen.
    """

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    total_kv_size_bytes: int = 0
    peak_memory_bytes: int = 0

    def record_hit(self) -> "CacheMetrics":
        """Return a new :class:`CacheMetrics` with hit incremented.

        Returns:
            CacheMetrics: Fresh metrics instance with ``hits``
            and ``total_requests`` each incremented by one.
        """
        return CacheMetrics(
            hits=self.hits + 1,
            misses=self.misses,
            total_requests=self.total_requests + 1,
            total_kv_size_bytes=self.total_kv_size_bytes,
            peak_memory_bytes=self.peak_memory_bytes,
        )

    def record_miss(self, kv_size_bytes: int = 0) -> "CacheMetrics":
        """Return a new :class:`CacheMetrics` with miss incremented.

        Args:
            kv_size_bytes: Size of the missed KV in bytes. The
                value is added to ``total_kv_size_bytes`` and
                ``peak_memory_bytes`` is updated if the new total
                is the largest observed so far.

        Returns:
            CacheMetrics: Fresh metrics instance with ``misses``
            and ``total_requests`` incremented and the size
            counters updated.
        """
        new_total = self.total_kv_size_bytes + kv_size_bytes
        return CacheMetrics(
            hits=self.hits,
            misses=self.misses + 1,
            total_requests=self.total_requests + 1,
            total_kv_size_bytes=new_total,
            # Peak is monotone — it tracks the largest cumulative
            # missed size ever seen.
            peak_memory_bytes=max(self.peak_memory_bytes, new_total),
        )

    def hit_rate(self) -> float:
        """Return current cache hit rate.

        Returns:
            float: ``hits / total_requests``, or ``0.0`` when no
            requests have been recorded.
        """
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def miss_rate(self) -> float:
        """Return current cache miss rate.

        Returns:
            float: ``misses / total_requests``, or ``0.0`` when
            no requests have been recorded.
        """
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests

    def size_growth_rate(self) -> float:
        """Return average KV size growth per miss in bytes.

        Returns:
            float: ``total_kv_size_bytes / misses``, or ``0.0``
            when no misses have been recorded.
        """
        if self.misses == 0:
            return 0.0
        return self.total_kv_size_bytes / self.misses
