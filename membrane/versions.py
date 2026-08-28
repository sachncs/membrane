"""PrefixVersionChain: maintain version lineage for incremental sync.

This module defines a small append-only version chain used by
:class:`~membrane.delta_sync.DeltaSync` and the canonical store to
track the evolution of a single prefix over time. Each entry records
its own ``version_id`` together with the ``parent_version`` it
descends from, allowing callers to walk the lineage and to compute
common ancestors for incremental merge operations.

Design rationale:
    The chain is *append-only*: once a version is appended it cannot
    be modified or removed. This makes the structure safe to share
    across threads without locks, and simplifies delta computation:
    two replicas can independently compute a common ancestor and
    then transfer only the diff between that ancestor and their
    current head.

    ``VersionEntry`` is a frozen dataclass, so the chain itself is
    hashable and can be safely used as the basis for additional
    indices.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionEntry:
    """A single version in a prefix lineage.

    Attributes:
        prefix_hash: Content hash that this version resolves to.
            Equal hashes across different versions indicate that
            no actual content change occurred.
        version_id: Unique identifier of this version. Strictly
            increasing per chain; gaps are permitted.
        parent_version: ``version_id`` of the immediate parent, or
            ``None`` for the root of the chain.
    """

    prefix_hash: str
    version_id: int
    parent_version: int | None


class PrefixVersionChain:
    """Incremental version chain for a single prefix.

    Supports append-only versioning and common-ancestor queries.

    Attributes:
        versions: Mapping from ``version_id`` to its
            :class:`VersionEntry`.
        next_version: Monotonic counter used when callers do not
            supply an explicit ``version_id`` to
            :meth:`append_version`.
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
            version_id: Optional explicit version id. When
                ``None``, the chain's internal counter is used
                (monotonically increasing).
            parent_version: Parent version id. May be ``None`` for
                a root entry.

        Returns:
            int: The assigned version id.

        Raises:
            ValueError: If ``version_id`` is already present in
                the chain. Append-only means re-using an id is
                rejected rather than silently overwritten.
        """
        vid = version_id if version_id is not None else self.next_version
        if vid in self.versions:
            raise ValueError(f"Version {vid} already exists")
        self.versions[vid] = VersionEntry(prefix_hash, vid, parent_version)
        # Always advance next_version past the just-appended id so
        # auto-numbered appends remain strictly monotonic.
        self.next_version = max(self.next_version, vid + 1)
        return vid

    def get_version(self, version_id: int) -> VersionEntry | None:
        """Return the entry for a given version id.

        Args:
            version_id: Version to look up.

        Returns:
            VersionEntry | None: The matching entry, or ``None``
            when no such version exists.
        """
        return self.versions.get(version_id)

    def latest_version(self, prefix_hash: str) -> int | None:
        """Return the highest version id that matches ``prefix_hash``.

        Args:
            prefix_hash: Content hash to match.

        Returns:
            int | None: Latest matching version id, or ``None`` if
            no version resolves to this hash.
        """
        best: int | None = None
        for vid, entry in self.versions.items():
            if entry.prefix_hash == prefix_hash and (best is None or vid > best):
                best = vid
        return best

    def get_common_ancestor(self, v1: int, v2: int) -> int | None:
        """Find the lowest common ancestor of two versions.

        Walks the ancestry chain of ``v1`` collecting every
        ancestor id, then walks ``v2``'s ancestry until a
        previously-seen id is found. The first such id is the
        deepest common ancestor reachable from ``v2``.

        Args:
            v1: First version id.
            v2: Second version id.

        Returns:
            int | None: Common ancestor version id, or ``None`` if
            no common ancestor exists in the chain.
        """
        # Collect every ancestor of v1 (including v1 itself).
        ancestors1: set[int] = set()
        cursor1: int | None = v1
        while cursor1 is not None:
            ancestors1.add(cursor1)
            entry = self.versions.get(cursor1)
            if entry is None:
                # Reached a parent that is not in the chain —
                # stop walking this branch.
                break
            cursor1 = entry.parent_version

        # Walk v2's ancestry until we hit an id that was also on
        # v1's path. The first match is the deepest common
        # ancestor (because we walk from the latest version
        # downward).
        cursor2: int | None = v2
        while cursor2 is not None:
            if cursor2 in ancestors1:
                return cursor2
            entry = self.versions.get(cursor2)
            if entry is None:
                break
            cursor2 = entry.parent_version

        return None
