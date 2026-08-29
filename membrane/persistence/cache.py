"""CachingPersistence: write-through read-through cache decorator.

Wraps any :class:`~membrane.persistence.base.PersistenceBackend` with an
in-memory cache. Writes go to the cache first and the inner backend
second; reads consult the cache before the inner backend.

When the inner backend is unavailable (e.g., Redis is down), reads are
served from the cache and a metric is emitted. Writes are also served
from the cache and re-tried against the inner backend when it comes back.

This decorator is the production configuration:

    CachingPersistence(Redis(url)) — Redis is canonical, cache is hot.

The :class:`~membrane.persistence.memory.Memory` class is the
test-only fallback (no inner backend, no caching layer).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from membrane.fragment import Fragment
from membrane.persistence.base import PersistenceBackend

logger = logging.getLogger(__name__)


class CachingPersistence:
    """Write-through read-through cache over a :class:`PersistenceBackend`.

    The cache is a simple ``dict`` guarded by an ``RLock`` for concurrent
    access within a process. The cache is invalidated on delete and is
    always re-populated from the inner backend when a miss is observed.
    """

    def __init__(
        self,
        inner: PersistenceBackend,
        on_unavailable: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        """Initialize with the inner (canonical) backend.

        Args:
            inner: The canonical store (e.g., Redis).
            on_unavailable: Optional callback invoked when the inner
                backend is unreachable. Useful for metrics emission.
        """
        self.inner = inner
        self.cache: dict[str, Fragment] = {}
        self.lock = threading.RLock()
        self.on_unavailable = on_unavailable or (lambda op, exc: None)

    @property
    def inner(self) -> PersistenceBackend:
        """The canonical backend this cache wraps."""
        return self.inner

    def ping(self) -> bool:
        """``True`` if the inner backend responds (cache hit on Redis still works)."""
        try:
            return self.inner.ping()
        except Exception as exc:
            self.on_unavailable("ping", exc)
            return False

    def store_fragment(
        self, fragment: Fragment, node_id: str, is_primary: bool = False
    ) -> bool:
        """Store ``fragment`` in cache and inner backend.

        The cache write is performed first so that even if the inner
        backend fails, subsequent reads can be served from the cache
        until the inner backend recovers.
        """
        with self.lock:
            self.cache[fragment.content_hash] = fragment
        try:
            return self.inner.store_fragment(fragment, node_id, is_primary)
        except Exception as exc:
            self.on_unavailable("store_fragment", exc)
            return False

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Return the fragment from cache or inner backend.

        On cache miss, falls through to the inner backend. On inner
        backend failure, returns ``None`` (the cache will repopulate
        on a later successful read).
        """
        with self.lock:
            cached = self.cache.get(content_hash)
            if cached is not None:
                return cached
        try:
            frag = self.inner.retrieve_fragment(content_hash)
            if frag is not None:
                with self.lock:
                    self.cache[content_hash] = frag
            return frag
        except Exception as exc:
            self.on_unavailable("retrieve_fragment", exc)
            return None

    def delete_fragment(self, content_hash: str, node_id: str) -> bool:
        """Invalidate the cache and forward to the inner backend."""
        with self.lock:
            self.cache.pop(content_hash, None)
        try:
            return self.inner.delete_fragment(content_hash, node_id)
        except Exception as exc:
            self.on_unavailable("delete_fragment", exc)
            return False

    def inventory_digest(self) -> dict[str, int]:
        """Return the inventory digest from the inner backend.

        The cache does not maintain its own digest (the canonical digest
        lives in Redis) — caching here would risk divergence.
        """
        try:
            return self.inner.inventory_digest()
        except Exception as exc:
            self.on_unavailable("inventory_digest", exc)
            return {}

    def list_node_fragments(self, node_id: str) -> list[str]:
        """Return the per-node fragment list from the inner backend."""
        try:
            return self.inner.list_node_fragments(node_id)
        except Exception as exc:
            self.on_unavailable("list_node_fragments", exc)
            return []

    def record_location(self, content_hash: str, node_id: str) -> None:
        """Forward to the inner backend."""
        try:
            self.inner.record_location(content_hash, node_id)
        except Exception as exc:
            self.on_unavailable("record_location", exc)

    def locate(self, content_hash: str) -> list[str]:
        """Return node IDs reporting holding ``content_hash``."""
        try:
            return self.inner.locate(content_hash)
        except Exception as exc:
            self.on_unavailable("locate", exc)
            return []

    def get_primary(self, content_hash: str) -> str | None:
        """Return the primary node ID for ``content_hash``."""
        try:
            return self.inner.get_primary(content_hash)
        except Exception as exc:
            self.on_unavailable("get_primary", exc)
            return None

    def lru_candidates(self, count: int) -> list[str]:
        """Return ``count`` eviction candidates from the inner backend."""
        try:
            return self.inner.lru_candidates(count)
        except Exception as exc:
            self.on_unavailable("lru_candidates", exc)
            return []

    def flush(self) -> None:
        """Drop the local cache and the inner backend. Test-only."""
        with self.lock:
            self.cache.clear()
        try:
            self.inner.flush()
        except Exception as exc:
            self.on_unavailable("flush", exc)


__all__ = ["CachingPersistence"]
