"""CacheMetrics: hit rate, size growth, and performance tracking."""

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
        total_requests: Total number of lookups.
        total_kv_size_bytes: Total size of missed KV in bytes.
        peak_memory_bytes: Peak cumulative size seen.
    """

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    total_kv_size_bytes: int = 0
    peak_memory_bytes: int = 0

    def record_hit(self) -> "CacheMetrics":
        """Return a new CacheMetrics with hit incremented."""
        return CacheMetrics(
            hits=self.hits + 1,
            misses=self.misses,
            total_requests=self.total_requests + 1,
            total_kv_size_bytes=self.total_kv_size_bytes,
            peak_memory_bytes=self.peak_memory_bytes,
        )

    def record_miss(self, kv_size_bytes: int = 0) -> "CacheMetrics":
        """Return a new CacheMetrics with miss incremented.

        Args:
            kv_size_bytes: Size of the missed KV in bytes.
        """
        new_total = self.total_kv_size_bytes + kv_size_bytes
        return CacheMetrics(
            hits=self.hits,
            misses=self.misses + 1,
            total_requests=self.total_requests + 1,
            total_kv_size_bytes=new_total,
            peak_memory_bytes=max(self.peak_memory_bytes, new_total),
        )

    def hit_rate(self) -> float:
        """Return current cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def miss_rate(self) -> float:
        """Return current cache miss rate."""
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests

    def size_growth_rate(self) -> float:
        """Return average KV size growth per miss in bytes."""
        if self.misses == 0:
            return 0.0
        return self.total_kv_size_bytes / self.misses
