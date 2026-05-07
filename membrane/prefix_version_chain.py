"""PrefixVersionChain: maintain version lineage for incremental sync."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionEntry:
    """A single version in a prefix lineage.

    Attributes:
        prefix_hash: Content hash at this version.
        version_id: Version identifier.
        parent_version: Previous version id (None for root).
    """

    prefix_hash: str
    version_id: int
    parent_version: int | None


class PrefixVersionChain:
    """Incremental version chain for a single prefix.

    Supports append-only versioning and common-ancestor queries.
    """

    def __init__(self) -> None:
        """Initialize an empty version chain."""
        self.versions: dict[int, VersionEntry] = {}
        self.next_version: int = 1

    def append_version(
        self,
        prefix_hash: str,
        version_id: int | None = None,
        parent_version: int | None = None,
    ) -> int:
        """Append a new version and return its id.

        Args:
            prefix_hash: Content hash for this version.
            version_id: Optional explicit version id.
            parent_version: Parent version id.

        Returns:
            The assigned version id.
        """
        vid = version_id if version_id is not None else self.next_version
        if vid in self.versions:
            raise ValueError(f"Version {vid} already exists")
        self.versions[vid] = VersionEntry(prefix_hash, vid, parent_version)
        self.next_version = max(self.next_version, vid + 1)
        return vid

    def get_version(self, version_id: int) -> VersionEntry | None:
        """Return the entry for a given version id.

        Args:
            version_id: Version to look up.

        Returns:
            VersionEntry if found, else None.
        """
        return self.versions.get(version_id)

    def latest_version(self, prefix_hash: str) -> int | None:
        """Return the highest version id that matches the given hash.

        Args:
            prefix_hash: Content hash to match.

        Returns:
            Latest matching version id, or None if no match.
        """
        best: int | None = None
        for vid, entry in self.versions.items():
            if entry.prefix_hash == prefix_hash:
                if best is None or vid > best:
                    best = vid
        return best

    def get_common_ancestor(self, v1: int, v2: int) -> int | None:
        """Find the lowest common ancestor of two versions.

        Args:
            v1: First version id.
            v2: Second version id.

        Returns:
            Common ancestor version id, or None if no common ancestor.
        """
        ancestors1: set[int] = set()
        cursor1: int | None = v1
        while cursor1 is not None:
            ancestors1.add(cursor1)
            entry = self.versions.get(cursor1)
            if entry is None:
                break
            cursor1 = entry.parent_version

        cursor2: int | None = v2
        while cursor2 is not None:
            if cursor2 in ancestors1:
                return cursor2
            entry = self.versions.get(cursor2)
            if entry is None:
                break
            cursor2 = entry.parent_version

        return None
