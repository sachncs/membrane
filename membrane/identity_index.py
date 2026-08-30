"""Identity-keyed fragment index.

The :class:`IdentityIndex` provides collision-safe lookups when two
fragments have the same ``identity.payload_hash`` but different
``layer_range``, ``head_range``, ``token_span``, ``dtype``, or
``shape`` — i.e. the store has multiple variants. The existing
:class:`~membrane.index.Index` always keys on the payload hash, which
silently merges them; :class:`IdentityIndex` keys on the full
:class:`~membrane.identity.PayloadIdentity` so two windows of the same
model at different layer/head ranges no longer alias.

The class is intentionally small and synchronous: it is a
``dict[identity_fingerprint -> content_hash]`` thread-safe via an
internal lock, plus the inverse ``dict[content_hash -> set[fingerprint]]``
that lets callers purge every identity variant when a fragment is
removed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from membrane.identity import PayloadIdentity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexEntry:
    """A single identity → content_hash binding inside :class:`IdentityIndex`.

    Attributes:
        content_hash: The fragment's payload hash (the wire-keyed
            primary identifier carried by every transport).
        identity: The full :class:`PayloadIdentity` the entry was
            keyed on.
    """

    content_hash: str
    identity: PayloadIdentity


class IdentityIndex:
    """Map :class:`PayloadIdentity` to ``content_hash`` for collision-safe lookup.

    Two fragments can share the same ``identity.payload_hash`` but
    differ in their model, layer range, head range, dtype, or
    shape — every existing in-memory index keys on the hash alone
    and so loses the distinction. :class:`IdentityIndex` keeps the
    full fingerprint so lookups return the exact binding the
    producer intended.

    Thread safety:
        All public methods are guarded by an internal
        :class:`threading.RLock` so the index can be shared across
        threads.

    Attributes:
        forward: :class:`dict[str, IndexEntry]`: ``identity.fingerprint()``
            → :class:`IndexEntry`.
        reverse: :class:`dict[str, set[str]]``: ``content_hash`` →
            ``{fingerprint, ...}`` so a delete can purge every
            identity variant stored against the same payload hash.
    """

    def __init__(self) -> None:
        """Initialize an empty index."""
        self.forward: dict[str, IndexEntry] = {}
        self.reverse: dict[str, set[str]] = {}
        self.lock = threading.RLock()

    def insert(self, identity: PayloadIdentity, content_hash: str) -> IndexEntry:
        """Bind an identity to a content hash, returning the entry.

        Idempotent: re-inserting the same fingerprint and content
        hash returns the existing entry untouched. Re-inserting the
        same fingerprint with a different content hash replaces
        the binding and logs a warning (this can only happen when
        a producer racily reused a fingerprint for unrelated bytes,
        which is a real bug).

        Args:
            identity: The full fingerprint.
            content_hash: The payload hash carried by the wire.

        Returns:
            IndexEntry: The stored binding.

        Raises:
            ValueError: If ``identity.payload_hash`` does not match
                ``content_hash`` (caller-mismatch detection).
        """
        if identity.payload_hash != content_hash:
            raise ValueError(
                f"identity.payload_hash={identity.payload_hash} disagrees with "
                f"supplied content_hash={content_hash}"
            )
        with self.lock:
            fp = identity.fingerprint()
            existing = self.forward.get(fp)
            if existing is not None and existing.content_hash != content_hash:
                logger.warning(
                    "Identity fingerprint collision: ref=%s existing=%s incoming=%s; rebinding",
                    fp,
                    existing.content_hash,
                    content_hash,
                )
            entry = IndexEntry(content_hash=content_hash, identity=identity)
            self.forward[fp] = entry
            self.reverse.setdefault(content_hash, set()).add(fp)
            return entry

    def lookup(self, identity: PayloadIdentity) -> IndexEntry | None:
        """Return the entry for ``identity`` or ``None`` when absent.

        Args:
            identity: Fingerprint to look up.

        Returns:
            IndexEntry | None: The stored entry or ``None``.
        """
        with self.lock:
            return self.forward.get(identity.fingerprint())

    def lookup_by_hash(self, content_hash: str) -> list[IndexEntry]:
        """Return every entry bound to ``content_hash``.

        Args:
            content_hash: Hash to look up.

        Returns:
            list[IndexEntry]: One entry per registered identity
            variant of the same payload hash. Empty when nothing
            has been bound.
        """
        with self.lock:
            return [
                self.forward[fp]
                for fp in sorted(self.reverse.get(content_hash, set()))
            ]

    def remove(self, identity: PayloadIdentity) -> bool:
        """Remove the binding for ``identity``.

        Args:
            identity: Fingerprint to remove.

        Returns:
            bool: ``True`` when a binding was actually removed.
        """
        with self.lock:
            fp = identity.fingerprint()
            entry = self.forward.pop(fp, None)
            if entry is None:
                return False
            reverse_set = self.reverse.get(entry.content_hash)
            if reverse_set is not None:
                reverse_set.discard(fp)
                if not reverse_set:
                    self.reverse.pop(entry.content_hash, None)
            return True

    def clear(self) -> None:
        """Drop every entry."""
        with self.lock:
            self.forward.clear()
            self.reverse.clear()

    def __len__(self) -> int:
        """Return the number of bound identities."""
        with self.lock:
            return len(self.forward)

    def __contains__(self, identity: object) -> bool:
        """Dunder ``__contains__`` for ``identity in index`` checks."""
        if not isinstance(identity, PayloadIdentity):
            return False
        with self.lock:
            return identity.fingerprint() in self.forward


__all__ = ["IdentityIndex", "IndexEntry"]
