"""The fragment table: resident fragments and their bookkeeping.

:class:`FragmentTable` owns the mutable state of one node's fragments:
the fragment map, insertion and access times, the primary set, and the
memory total. :class:`~membrane.node.Node` composes it with an index,
a graph, an eviction policy, and a tenant guard.

Writes take the table's re-entrant lock, so callers that need several
operations to be atomic can hold ``table.lock`` around them. Single-key
reads (:meth:`get`, ``in``, :meth:`inserted_at`) do not: one dict
operation is atomic (and, on a free-threaded Python, internally
synchronized), so the read path scales across threads without serializing
on the lock. A reader may see a fragment that a concurrent writer is about
to remove, exactly as if the read had happened first.

:meth:`record_access` writes to a per-thread buffer rather than the shared
``access_times`` map, so concurrent reads do not contend on one dict.
Buffers are merged into ``access_times`` whenever it is read through
:meth:`recency` (eviction, snapshots). A read recorded concurrently with a
merge can be lost; recency is a heuristic, so that is harmless.
"""

import threading

from membrane.fragment import Fragment
from membrane.runtime.concurrency import share


class FragmentTable:
    """Resident fragments of one node.

    Attributes:
        fragments: ``content_hash -> Fragment``.
        primary_hashes: Hashes this node owns as primary.
        access_times: ``content_hash ->`` last access (Unix time).
        insertion_times: ``content_hash ->`` insertion (Unix time).
        memory_usage: Sum of resident ``payload_size`` in bytes.
        lock: Re-entrant lock guarding every field.
    """

    def __init__(self) -> None:
        """Create an empty table."""
        self.fragments: dict[str, Fragment] = {}
        self.primary_hashes: set[str] = set()
        self.access_times: dict[str, float] = {}
        self.insertion_times: dict[str, float] = {}
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.__local = threading.local()
        self.__buffers: list[dict[str, float]] = []
        share(self.__local)

    def __contains__(self, content_hash: object) -> bool:
        """Whether ``content_hash`` is resident.

        Args:
            content_hash: Content hash.

        Returns:
            bool: True when resident.
        """
        return content_hash in self.fragments

    def __len__(self) -> int:
        """Number of resident fragments.

        Returns:
            int: The fragment count.
        """
        return len(self.fragments)

    def get(self, content_hash: str) -> Fragment | None:
        """Return a resident fragment (no lock: one dict read).

        Args:
            content_hash: Content hash.

        Returns:
            Fragment | None: The fragment, or ``None`` when absent.
        """
        return self.fragments.get(content_hash)

    def inserted_at(self, content_hash: str, default: float) -> float:
        """Return when a fragment was inserted (no lock: one dict read).

        Args:
            content_hash: Content hash.
            default: Returned when unknown.

        Returns:
            float: Insertion time.
        """
        return self.insertion_times.get(content_hash, default)

    def record_access(self, content_hash: str, now: float) -> None:
        """Record a read in this thread's buffer (no lock, no shared write).

        Args:
            content_hash: Content hash.
            now: Access time.
        """
        buffer = getattr(self.__local, "accesses", None)
        if buffer is None:
            buffer = self.__local.accesses = {}
            with self.lock:
                self.__buffers.append(buffer)
        buffer[content_hash] = now

    def recency(self) -> dict[str, float]:
        """Merge every thread's buffered reads and return ``access_times``.

        Returns:
            dict[str, float]: Last access per resident hash.
        """
        with self.lock:
            for buffer in self.__buffers:
                pending = buffer.copy()
                buffer.clear()
                for content_hash, at in pending.items():
                    if content_hash in self.fragments and at > self.access_times.get(content_hash, 0.0):
                        self.access_times[content_hash] = at
            return self.access_times

    def add(self, fragment: Fragment, now: float) -> None:
        """Insert a new fragment (the caller has checked capacity).

        Args:
            fragment: The fragment.
            now: Insertion time.
        """
        content_hash = fragment.identity.payload_hash
        with self.lock:
            self.fragments[content_hash] = fragment
            self.memory_usage += fragment.payload_size
            self.insertion_times[content_hash] = now

    def touch(self, content_hash: str, now: float, is_primary: bool = False) -> None:
        """Record an access, and optionally primary ownership.

        Args:
            content_hash: Content hash.
            now: Access time.
            is_primary: Mark the fragment as owned by this node.
        """
        with self.lock:
            self.access_times[content_hash] = now
            if is_primary:
                self.primary_hashes.add(content_hash)

    def pop(self, content_hash: str) -> Fragment:
        """Remove a fragment and all its bookkeeping.

        Args:
            content_hash: Content hash of a resident fragment.

        Returns:
            Fragment: The removed fragment.

        Raises:
            KeyError: When the fragment is not resident.
        """
        with self.lock:
            fragment = self.fragments.pop(content_hash)
            self.memory_usage -= fragment.payload_size
            self.primary_hashes.discard(content_hash)
            self.access_times.pop(content_hash, None)
            self.insertion_times.pop(content_hash, None)
            return fragment

    def is_expired(self, content_hash: str, now: float) -> bool:
        """Whether a resident fragment has outlived its TTL.

        Args:
            content_hash: Content hash.
            now: Current time.

        Returns:
            bool: True when its age exceeds its ``ttl``; False when absent.
        """
        with self.lock:
            fragment = self.fragments.get(content_hash)
            if fragment is None:
                return False
            return now - self.insertion_times.get(content_hash, now) > fragment.ttl

    def expired(self, now: float) -> list[str]:
        """Return every resident hash past its TTL.

        Args:
            now: Current time.

        Returns:
            list[str]: Expired content hashes.
        """
        with self.lock:
            return [h for h in self.fragments if self.is_expired(h, now)]

    def snapshot(self) -> dict[str, Fragment]:
        """Return a consistent copy of the fragment map.

        Returns:
            dict[str, Fragment]: ``content_hash -> fragment`` at one instant.
        """
        with self.lock:
            return dict(self.fragments)

    def primaries(self) -> set[str]:
        """Return a copy of the primary set.

        Returns:
            set[str]: Hashes owned as primary.
        """
        with self.lock:
            return set(self.primary_hashes)


__all__ = ["FragmentTable"]
