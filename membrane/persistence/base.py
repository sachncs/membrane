"""PersistenceBackend: polymorphic interface for fragment persistence.

Three concrete backends are provided:

* :class:`~membrane.persistence.memory.Memory` — in-process dict-backed
  implementation. Used as test-only fallback; in production it is wrapped
  in :class:`CachingPersistence` as a read-through cache over Redis.
* :class:`~membrane.persistence.redis.Redis` — Redis-backed canonical store.
* :class:`CachingPersistence` — decorator that adds a write-through
  in-memory cache over any inner backend.

The persistence layer is independent of transport and cluster — the same
backend instance can be shared across processes and threads within a node.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membrane.fragment import Fragment


@runtime_checkable
class PersistenceBackend(Protocol):
    """Protocol every persistence backend must implement.

    The interface is intentionally small: stores do not need to be
    thread-safe (callers serialize access via ``MembraneNode``), but they
    must be safe to share across coroutines within a single thread.
    """

    def ping(self) -> bool:
        """Return ``True`` if the backend is reachable."""
        ...

    def store_fragment(self, fragment: Fragment, node_id: str, is_primary: bool = False) -> bool:
        """Persist ``fragment`` owned by ``node_id``.

        Returns:
            bool: True on success, False if the backend is unreachable.
        """
        ...

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Fetch a fragment by its content hash, or ``None`` if absent."""
        ...

    def delete_fragment(self, content_hash: str, node_id: str) -> bool:
        """Remove a fragment owned by ``node_id``.

        Returns:
            bool: True if removed, False otherwise.
        """
        ...

    def inventory_digest(self) -> dict[str, int]:
        """Return a ``content_hash -> version_id`` map of every stored fragment."""
        ...

    def list_node_fragments(self, node_id: str) -> list[str]:
        """Return content hashes owned by ``node_id``."""
        ...

    def record_location(self, content_hash: str, node_id: str) -> None:
        """Record that ``node_id`` holds ``content_hash``."""
        ...

    def locate(self, content_hash: str) -> list[str]:
        """Return the set of node IDs that report holding ``content_hash``."""
        ...

    def get_primary(self, content_hash: str) -> str | None:
        """Return the primary node ID for ``content_hash``, if any."""
        ...

    def lru_candidates(self, count: int) -> list[str]:
        """Return ``count`` content hashes eligible for LRU eviction."""
        ...

    def flush(self) -> None:
        """Drop every entry. Test-only."""
        ...


__all__ = ["PersistenceBackend"]
