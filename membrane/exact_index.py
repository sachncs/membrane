"""Exact index: content_hash -> Fragment + location set."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass(frozen=True)
class IndexEntry:
    """Entry stored in the ExactIndex."""

    fragment: Fragment
    locations: frozenset[str]


class ExactIndex:
    """In-memory exact index keyed by content_hash.

    .. note::
        This class is **not thread-safe**.  The internal ``entries`` dict
        is not protected by locks.  If the index is accessed from
        multiple threads, the caller must provide external
        synchronisation.
    """

    def __init__(self) -> None:
        self.entries: dict[str, IndexEntry] = {}

    def insert(self, fragment: Fragment, locations: set[str]) -> None:
        """Insert or overwrite a fragment and its locations.

        Args:
            fragment: The fragment to index.
            locations: Set of node IDs holding the fragment.
        """
        self.entries[fragment.content_hash] = IndexEntry(
            fragment=fragment, locations=frozenset(locations)
        )

    def lookup(self, content_hash: str) -> IndexEntry | None:
        """Look up a fragment by its content hash.

        Args:
            content_hash: Hash to look up.

        Returns:
            IndexEntry if found, else None.
        """
        return self.entries.get(content_hash)

    def add_location(self, content_hash: str, node_id: str) -> bool:
        """Add a location to an existing entry.

        Args:
            content_hash: Hash of the fragment.
            node_id: Node ID to add.

        Returns:
            True if added, False if entry did not exist.
        """
        entry = self.entries.get(content_hash)
        if entry is None:
            return False
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
            True if the fragment was present and removed.
        """
        if content_hash in self.entries:
            del self.entries[content_hash]
            return True
        return False
