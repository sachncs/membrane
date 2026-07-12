"""KVCacheManager: dedicated cache layer with hit/miss tracking.

This module defines :class:`KVCacheManager`, a single-region
memory cache that bundles a
:class:`~membrane.index_system.IndexSystem`, an LRU eviction
front, and a :class:`~membrane.cache_metrics.CacheMetrics`
tracker.

The cache layer separates three concerns:

* **Prefill** — long context processing lives elsewhere (see
  :class:`~membrane.prefill_adapter.PrefillAdapter`); the cache
  only stores the resulting fragments.
* **Decode** — request-time decoding reads fragments via
  :meth:`lookup_kv`.
* **Cache** — the manager itself, with explicit hit/miss
  tracking.

The manager maintains a ``prefix_to_fragments`` mapping so that
lookups by prefix hash can return the full list of fragments that
were stored together for that prefix. When ``max_prefixes`` is
set, the LRU eviction removes the least-recently-accessed prefix
on insert overflow.
"""

import logging

logger = logging.getLogger(__name__)

from membrane.cache_metrics import CacheMetrics
from membrane.fragment import Fragment
from membrane.index_system import IndexSystem
from membrane.lru_cache import LRUCache


class KVCacheManager:
    """Single-region memory cache that separates prefill, decode, and cache.

    Tracks cache hit rates and routes lookups through an owned
    :class:`IndexSystem`. Maintains a mapping from prefix hash to
    the fragment hashes that were stored together so that lookup
    by prefix works correctly.

    When ``max_prefixes`` is set, LRU eviction removes the
    least-recently-accessed prefix on insert overflow.

    Attributes:
        index_system: Backing index system.
        metrics: Cache hit/miss statistics.
        prefix_to_fragments: Mapping from prefix hash to the
            list of fragment hashes stored for that prefix.
        lru: :class:`LRUCache` used to bound ``prefix_to_fragments``.
        max_prefixes: Configured upper bound on prefixes
            (``None`` for unbounded).
    """

    def __init__(
        self,
        index_system: IndexSystem | None = None,
        max_prefixes: int | None = None,
    ) -> None:
        """Initialize with an optional index system.

        Args:
            index_system: Index system backing the cache. A
                fresh one is created when ``None``.
            max_prefixes: Maximum number of prefix entries to
                retain. ``None`` means unbounded (LRU tracking
                remains active but no eviction is triggered).
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
        """Store KV fragments keyed by ``prefix_hash``.

        Each fragment is registered in the index system with
        ``node_id`` as the sole replica holder; the prefix entry
        records the fragment hashes in insertion order. After the
        insert, any prefix evicted by the LRU is removed via
        :meth:`remove_prefix`.

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
        # Evict *after* touching the new prefix so the just-
        # inserted entry cannot be immediately evicted.
        for evicted in self.lru.evict_if_over():
            self.remove_prefix(evicted)
        logger.debug(
            "Stored %d fragments for prefix %s", len(kv_fragments), prefix_hash
        )

    def lookup_kv(self, prefix_hash: str) -> list[Fragment]:
        """Look up fragments by prefix hash with hit/miss tracking.

        On hit, the LRU is touched and the metrics recorder is
        updated with a hit. On miss, the metrics recorder is
        updated with a miss (with ``kv_size_bytes=0`` because the
        caller can supply that separately).

        Args:
            prefix_hash: Hash to look up.

        Returns:
            list[Fragment]: List of fragments if cache hit,
            otherwise empty. A hit may still return fewer
            fragments than were stored if some were evicted from
            the underlying index.
        """
        fragment_hashes = self.prefix_to_fragments.get(prefix_hash)
        if fragment_hashes is not None:
            # Hit: refresh LRU and metrics.
            self.metrics = self.metrics.record_hit()
            self.lru.touch(prefix_hash)
            result: list[Fragment] = []
            for h in fragment_hashes:
                entry = self.index_system.exact_lookup(h)
                if entry is not None:
                    result.append(entry.fragment)
            return result
        # Miss: update metrics and return empty.
        self.metrics = self.metrics.record_miss(kv_size_bytes=0)
        return []

    def remove_prefix(self, prefix_hash: str) -> bool:
        """Remove a prefix entry and its fragment mappings.

        Only the prefix-to-fragments mapping and the LRU entry
        are removed; the fragments themselves remain in the index
        system because other prefixes may still reference them.

        Args:
            prefix_hash: Prefix hash to remove.

        Returns:
            bool: True if the prefix existed and was removed,
            False otherwise.
        """
        if prefix_hash in self.prefix_to_fragments:
            del self.prefix_to_fragments[prefix_hash]
            self.lru.remove(prefix_hash)
            logger.debug("Removed prefix %s", prefix_hash)
            return True
        return False

    def get_hit_rate(self) -> float:
        """Return current cache hit rate.

        Returns:
            float: Hit rate in ``[0, 1]``.
        """
        return self.metrics.hit_rate()

    def get_miss_rate(self) -> float:
        """Return current cache miss rate.

        Returns:
            float: Miss rate in ``[0, 1]``.
        """
        return self.metrics.miss_rate()

    def get_metrics(self) -> CacheMetrics:
        """Return current cache metrics snapshot.

        Returns:
            CacheMetrics: The current immutable metrics value.
        """
        return self.metrics
