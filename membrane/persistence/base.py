"""Storage and Inventory: split persistence interfaces.

Historically the persistence layer was one big
:class:`PersistenceBackend` Protocol covering both
fragment CRUD (per-hash ``store`` / ``retrieve`` /
``delete``) and cross-node directory queries
(``inventory_digest`` / ``primary`` / ``locate`` /
``lru_candidates``). The two concerns have very
different semantics (CRUD on a single fragment vs.
cluster-wide inventory) and different invariants, so
they live in their own Protocols:

* :class:`Storage` — per-node CRUD over a single
  fragment by hash. ``Memory`` and ``Redis`` both
  provide this; :class:`CachingPersistence` decorates
  any :class:`Storage`.
* :class:`Inventory` — cross-node directory queries.
  ``Memory`` and ``Redis`` both provide this; the two
  are independent of :class:`Storage`'s
  read/write path.

Methods that exercised both (such as ``ping``) remain
on both Protocols. ``delete_fragment`` was previously
declared with a ``node_id`` argument even though
neither concrete backend needed it; the Storage
Protocol uses a single-argument form that matches the
concrete implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membrane.fragment import Fragment


@runtime_checkable
class Storage(Protocol):
    """Per-fragment CRUD operations on a persistence backend.

    Implementations are not required to be thread-safe;
    callers (such as :class:`~membrane.node.Node`)
    serialize access via their own locks. The interface
    is suitable for sharing across coroutines within
    one thread.
    """

    def ping(self) -> bool:
        """Return ``True`` if the backend is reachable."""
        ...

    def store_fragment(self, fragment: Fragment, node_id: str, is_primary: bool = False) -> bool:
        """Persist ``fragment`` owned by ``node_id``."""
        ...

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Fetch a fragment by ``content_hash``, or ``None`` if absent."""
        ...

    def delete_fragment(self, content_hash: str) -> bool:
        """Remove the fragment with the given ``content_hash``.

        Args:
            content_hash: Content-addressed hash of the
                fragment to remove.

        Returns:
            bool: True if the fragment was removed,
            False if no fragment with that hash existed.
        """
        ...

    def list_node_fragments(self, node_id: str) -> list[str]:
        """Return content hashes owned by ``node_id``."""
        ...

    def flush(self) -> None:
        """Drop every entry. Test-only."""
        ...


@runtime_checkable
class Inventory(Protocol):
    """Cross-node fragment-location queries.

    This is the directory-style surface — every
    fragment maps to the set of nodes that report
    holding it, plus a primary owner selected by some
    rule. Backends can implement Storage alone, or
    Inventory alone, or both.
    """

    def ping(self) -> bool:
        """Return ``True`` if the backend is reachable."""
        ...

    def inventory_digest(self) -> dict[str, int]:
        """``content_hash -> version_id`` for every stored fragment."""
        ...

    def record_location(self, content_hash: str, node_id: str) -> None:
        """Record that ``node_id`` holds ``content_hash``."""
        ...

    def locate(self, content_hash: str) -> list[str]:
        """Node IDs that report holding ``content_hash``."""
        ...

    def get_primary(self, content_hash: str) -> str | None:
        """Primary node ID for ``content_hash``, if any."""
        ...

    def lru_candidates(self, count: int) -> list[str]:
        """``count`` content hashes eligible for LRU eviction."""
        ...


# Backwards-compatibility alias used by tests and external
# callers. The previous monolithic Protocol remains
# importable so deep-imports continue to work; the alias
# re-exports both storage and inventory query methods so
# existing ``isinstance(x, PersistenceBackend)`` checks
# keep matching Memory / Redis. New code should depend on
# the specific protocol it needs.
class PersistenceBackend(Storage, Inventory, Protocol):
    """Combined storage + inventory protocol.

    Historical alias covering every persistence
    operation. New code should depend on :class:`Storage`
    or :class:`Inventory` directly.
    """


__all__ = ["Inventory", "PersistenceBackend", "Storage"]
