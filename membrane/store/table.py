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

The read path (:meth:`lookup`) goes through a per-thread cache. On a
free-threaded Python, looking up a key string the request just parsed in
the shared map takes a reference to the stored key for the comparison,
and those reference-count updates on shared objects stop reads from
scaling with cores. Each thread's cache is keyed by its own strings, so
a hit touches only that thread's objects. :meth:`pop` removes the key from
every cache, and a fill that races with a removal is undone. Mutate the
table only through :meth:`add` and :meth:`pop`, or caches go stale.

Reads are recorded in the same per-thread state rather than the shared
``access_times`` map, and merged into it whenever it is read through
:meth:`recency` (eviction, snapshots). A read recorded concurrently with a
merge can be lost; recency is a heuristic, so that is harmless.
"""

import threading
import weakref

from membrane.fragment import Fragment
from membrane.runtime.concurrency import share
from membrane.store.digest import InventoryDigest

#: Entries one thread's read cache holds before it starts over.
READ_CACHE_ENTRIES = 1 << 16


class ReadCache(dict[str, tuple[Fragment, float]]):
    """One thread's ``key -> (fragment, expiry)`` cache (see :meth:`FragmentTable.lookup`).

    Attributes:
        accesses: The thread's reads not yet merged into ``access_times``.
    """

    __slots__ = ("__weakref__", "accesses")

    def __init__(self) -> None:
        """Start empty."""
        super().__init__()
        self.accesses: dict[str, float] = {}

    # Identity, so a WeakSet can hold the caches of every thread.
    __hash__ = object.__hash__  # type: ignore[assignment]
    __eq__ = object.__eq__  # type: ignore[assignment]


class FragmentTable:
    """Resident fragments of one node.

    Attributes:
        fragments: ``key -> Fragment``, keyed by
            :attr:`~membrane.fragment.Fragment.key` (the bare content hash
            for the default tenant, ``"tenant:hash"`` otherwise).
        primary_hashes: Hashes this node owns as primary.
        access_times: ``content_hash ->`` last access (Unix time).
        insertion_times: ``content_hash ->`` insertion (Unix time).
        memory_usage: Sum of resident ``payload_size`` in bytes.
        lock: Re-entrant lock guarding every field.
        digest: Bucketed inventory digest, updated on every add and pop.
        tenant_keys: ``content_hash ->`` keys of the resident copies that
            belong to a non-default tenant.
        payload_refs: ``payload_ref ->`` resident fragments referencing it
            (tenants storing identical content share one blob).
    """

    def __init__(self) -> None:
        """Create an empty table."""
        self.fragments: dict[str, Fragment] = {}
        self.primary_hashes: set[str] = set()
        self.access_times: dict[str, float] = {}
        self.insertion_times: dict[str, float] = {}
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.digest = InventoryDigest()
        self.tenant_keys: dict[str, set[str]] = {}
        self.payload_refs: dict[str, int] = {}
        self.__local = threading.local()
        self.__caches: weakref.WeakSet[ReadCache] = weakref.WeakSet()
        # Every thread's unmerged reads, kept after the thread exits until merged.
        self.__buffers: list[tuple[weakref.ref[ReadCache], dict[str, float]]] = []
        self.__removals = 0
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

    def thread_cache(self) -> ReadCache:
        """This thread's read cache (created on first use).

        Returns:
            ReadCache: The cache.
        """
        try:
            return self.__local.reads  # type: ignore[no-any-return]
        except AttributeError:
            cache = self.__local.reads = ReadCache()
            with self.lock:
                self.__caches.add(cache)
                self.__buffers.append((weakref.ref(cache), cache.accesses))
            return cache

    def lookup(self, content_hash: str, now: float | None = None) -> tuple[Fragment, float] | None:
        """Return a resident fragment and when it expires, through this thread's cache.

        On a free-threaded Python, a lookup in the shared ``fragments`` map
        with a key string the caller just built (every request parses its
        own) briefly takes a reference to the stored key, and the threads'
        reference-count updates on shared keys keep them from scaling. Each
        thread therefore keeps its own cache, keyed by its own strings.
        :meth:`pop` drops a key from every cache, and a fill that races
        with a removal is undone, so a cache never serves a removed fragment.

        Args:
            content_hash: Storage key.
            now: When given, a hit is also recorded as a read at this time
                (as :meth:`record_access` does).

        Returns:
            tuple[Fragment, float] | None: The fragment and its expiry time
            (insertion time plus TTL), or ``None`` when absent.
        """
        try:
            cache = self.__local.reads
        except AttributeError:
            cache = self.thread_cache()
        entry = cache.get(content_hash)
        if entry is not None:
            if now is not None:
                cache.accesses[content_hash] = now
            return entry
        removals = self.__removals
        fragment = self.fragments.get(content_hash)
        if fragment is None:
            return None
        entry = (fragment, self.insertion_times.get(content_hash, 0.0) + fragment.ttl)
        if len(cache) >= READ_CACHE_ENTRIES:
            cache.clear()
        cache[content_hash] = entry
        if self.__removals != removals:
            cache.pop(content_hash, None)
        if now is not None:
            cache.accesses[content_hash] = now
        return entry

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
        self.thread_cache().accesses[content_hash] = now

    def recency(self) -> dict[str, float]:
        """Merge every thread's buffered reads and return ``access_times``.

        Returns:
            dict[str, float]: Last access per resident hash.
        """
        with self.lock:
            # Buffers of exited threads are dropped once merged.
            buffers, self.__buffers = self.__buffers, []
            for owner, buffer in buffers:
                pending = buffer.copy()
                buffer.clear()
                if owner() is not None:
                    self.__buffers.append((owner, buffer))
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
        content_hash = fragment.key
        with self.lock:
            self.fragments[content_hash] = fragment
            self.memory_usage += fragment.payload_size
            self.insertion_times[content_hash] = now
            self.digest.add(content_hash, fragment.version_id)
            if content_hash != fragment.identity.payload_hash:
                self.tenant_keys.setdefault(fragment.identity.payload_hash, set()).add(content_hash)
            if fragment.payload_ref is not None:
                self.payload_refs[fragment.payload_ref] = self.payload_refs.get(fragment.payload_ref, 0) + 1

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
            self.__removals += 1
            for cache in self.__caches:
                cache.pop(content_hash, None)
            self.digest.remove(content_hash)
            self.memory_usage -= fragment.payload_size
            self.primary_hashes.discard(content_hash)
            self.access_times.pop(content_hash, None)
            self.insertion_times.pop(content_hash, None)
            copies = self.tenant_keys.get(fragment.identity.payload_hash)
            if copies is not None:
                copies.discard(content_hash)
                if not copies:
                    del self.tenant_keys[fragment.identity.payload_hash]
            ref = fragment.payload_ref
            if ref is not None:
                remaining = self.payload_refs.get(ref, 1) - 1
                if remaining > 0:
                    self.payload_refs[ref] = remaining
                else:
                    self.payload_refs.pop(ref, None)
            return fragment

    def references(self, payload_ref: str) -> int:
        """How many resident fragments reference a payload blob.

        Args:
            payload_ref: Content-store key.

        Returns:
            int: The count (0 when no resident fragment uses it).
        """
        return self.payload_refs.get(payload_ref, 0)

    def any_tenant_copy(self, payload_hash: str) -> str | None:
        """The key of some non-default tenant's copy of ``payload_hash``.

        Args:
            payload_hash: Content hash.

        Returns:
            str | None: A resident key, or ``None`` when no other tenant
            holds the content.
        """
        with self.lock:
            copies = self.tenant_keys.get(payload_hash)
            return min(copies) if copies else None

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
