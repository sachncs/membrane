"""Exact index: content_hash -> Fragment + location set.

This module implements :class:`ExactIndex`, the simplest of the
four in-memory lookup structures that ship with Membrane. It maps a
fragment's ``content_hash`` to the fragment itself together with the
set of node IDs currently holding a replica of it.

The exact index is the source of truth for *authoritative* lookups:
if a hash is present here, the system can resolve the underlying
fragment metadata without consulting remote nodes. Semantic,
positional, and co-access indexes are layered on top to answer
*approximate* queries that would otherwise require a scan.

Thread safety:
    The class is **not thread-safe** — it provides no internal
    locking and concurrent writers can corrupt the entries dict.
    The expected usage pattern is one writer thread plus multiple
    readers, or wrapping the instance in an external
    ``threading.RLock``. Tests relying on concurrent access should
    use the thread-safety suite under
    ``tests/membrane/test_thread_safety.py``.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass(frozen=True)
class IndexEntry:
    """Entry stored in the :class:`ExactIndex`.

    The combination of a fragment and the set of nodes that hold a
    replica is sufficient to answer both "do I have the metadata?"
    and "who can serve it?" queries.

    Attributes:
        fragment: The fragment metadata itself.
        locations: Frozen set of node IDs that hold a replica.
            Stored as a ``frozenset`` so the entry itself is
            hashable and can participate in cached derived
            structures.
    """

    fragment: Fragment
    locations: frozenset[str]


class ExactIndex:
    """In-memory exact index keyed by ``content_hash``.

    .. note::
        This class is **not thread-safe**.  The internal ``entries``
        dict is not protected by locks.  If the index is accessed
        from multiple threads, the caller must provide external
        synchronisation.

    The index is intentionally minimal — it does not store the
    fragment payload. The payload lives in the
    :class:`~membrane.fragment_store.FragmentStore`; this index
    only records *where* the metadata and payload live.
    """

    def __init__(self) -> None:
        """Initialize an empty exact index."""
        self.entries: dict[str, IndexEntry] = {}

    def insert(self, fragment: Fragment, locations: set[str]) -> None:
        """Insert or overwrite a fragment and its locations.

        Insertion is *upsert*: if a fragment with the same
        ``content_hash`` already exists, it is replaced.

        Args:
            fragment: The fragment to index.
            locations: Set of node IDs holding the fragment. The
                set is converted to a ``frozenset`` for immutability.
        """
        self.entries[fragment.content_hash] = IndexEntry(
            fragment=fragment, locations=frozenset(locations)
        )

    def lookup(self, content_hash: str) -> IndexEntry | None:
        """Look up a fragment by its content hash.

        Args:
            content_hash: Hash to look up.

        Returns:
            IndexEntry | None: The matching entry, or ``None`` if no
            fragment with that hash is indexed.
        """
        return self.entries.get(content_hash)

    def add_location(self, content_hash: str, node_id: str) -> bool:
        """Add a location to an existing entry.

        Creates a new :class:`IndexEntry` with the updated location
        set rather than mutating the existing one (which is frozen).

        Args:
            content_hash: Hash of the fragment.
            node_id: Node ID to add.

        Returns:
            bool: True if the entry existed and ``node_id`` was
            added, False if the entry was not found.
        """
        entry = self.entries.get(content_hash)
        if entry is None:
            return False
        # Frozen dataclasses cannot be mutated in place; rebuild
        # with the augmented location set.
        new_locations = frozenset(entry.locations | {node_id})
        self.entries[content_hash] = IndexEntry(
            fragment=entry.fragment, locations=new_locations
        )
        return True

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from the index.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            bool: True if the fragment was present and removed,
            False otherwise.
        """
        if content_hash in self.entries:
            del self.entries[content_hash]
            return True
        return False
