"""KVCacheManager: dedicated cache layer with hit/miss tracking."""

import logging

logger = logging.getLogger(__name__)

from membrane.cache_metrics import CacheMetrics
from membrane.fragment import Fragment
from membrane.index_system import IndexSystem
from membrane.lru_cache import LRUCache


class KVCacheManager:
    """Single-region memory cache that separates prefill, decode, and cache.

    Tracks cache hit rates and routes lookups through an owned IndexSystem.
    Maintains a mapping from prefix_hash to the fragment hashes that were
    stored together, so that lookup by prefix works correctly.

    When *max_prefixes* is set, LRU eviction removes the least-recently-
    accessed prefix on insert overflow.
    """

    def __init__(
        self,
        index_system: IndexSystem | None = None,
        max_prefixes: int | None = None,
    ) -> None:
        """Initialize with an optional index system.

        Args:
            index_system: Index system backing the cache. Creates one if None.
            max_prefixes: Maximum number of prefix entries to retain.
        """
        self.index_system = index_system or IndexSystem()
        self.metrics = CacheMetrics()
        self.prefix_to_fragments: dict[str, list[str]] = {}
        self.lru = LRUCache(capacity=max_prefixes)
        self.max_prefixes = max_prefixes
        logger.info("Initialized %s", self.__class__.__name__)

    def store_kv(
        self,
        prefix_hash: str,
        kv_fragments: list[Fragment],
        node_id: str = "local",
    ) -> None:
        """Store KV fragments keyed by prefix hash.

        Args:
            prefix_hash: Hash of the token prefix.
            kv_fragments: Fragments representing KV segments.
            node_id: Node holding the fragments.
        """
        fragment_hashes: list[str] = []
        for frag in kv_fragments:
            self.index_system.insert(frag, {node_id})
            fragment_hashes.append(frag.content_hash)
        self.prefix_to_fragments[prefix_hash] = fragment_hashes
        self.lru.touch(prefix_hash)
        for evicted in self.lru.evict_if_over():
            self.remove_prefix(evicted)
        logger.debug(
            "Stored %d fragments for prefix %s", len(kv_fragments), prefix_hash
        )

    def lookup_kv(self, prefix_hash: str) -> list[Fragment]:
        """Lookup fragments by prefix hash with hit/miss tracking.

        Args:
            prefix_hash: Hash to look up.

        Returns:
            List of fragments if cache hit, else empty list.
        """
        fragment_hashes = self.prefix_to_fragments.get(prefix_hash)
        if fragment_hashes is not None:
            self.metrics = self.metrics.record_hit()
            self.lru.touch(prefix_hash)
            result: list[Fragment] = []
            for h in fragment_hashes:
                entry = self.index_system.exact_lookup(h)
                if entry is not None:
                    result.append(entry.fragment)
            return result
        self.metrics = self.metrics.record_miss(kv_size_bytes=0)
        return []

    def remove_prefix(self, prefix_hash: str) -> bool:
        """Remove a prefix entry and its fragment mappings.

        Args:
            prefix_hash: Prefix hash to remove.

        Returns:
            True if the prefix existed and was removed.
        """
        if prefix_hash in self.prefix_to_fragments:
            del self.prefix_to_fragments[prefix_hash]
            self.lru.remove(prefix_hash)
            logger.debug("Removed prefix %s", prefix_hash)
            return True
        return False

    def get_hit_rate(self) -> float:
        """Return current cache hit rate."""
        return self.metrics.hit_rate()

    def get_miss_rate(self) -> float:
        """Return current cache miss rate."""
        return self.metrics.miss_rate()

    def get_metrics(self) -> CacheMetrics:
        """Return current cache metrics snapshot."""
        return self.metrics
