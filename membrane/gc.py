"""Garbage collection primitives: ref-counts, tombstones, sweeper.

Memcached taught the industry that open-ended ``delete`` leads to
phantom reads: an active consumer still mid-flight when the
producer invalidates a key receives a stale value because
replication has not yet propagated. Membrane avoids this with
two durable primitives:

* :class:`RefCount` — a per-content-hash counter whose final
  decrement decides when the canonical bytes can really go away.
* :class:`Tombstone` — a soft-delete marker with a deadline
  (``until_unix_time``). A delete writes a tombstone first, lets
  gossip propagate, then becomes a hard delete only after the
  deadline (typically ``> 2 * gossip_convergence``).
* :class:`Sweeper` — a daemon thread that scans every
  ``DEFAULT_TTL_SWEEP_INTERVAL`` seconds for expired entries in
  both tables plus the local in-memory fragments.

The three together replace the prior opportunistic ``Node.evict``
hooks (which fired only on the read path) and the pre-existing
``DEFAULT_TTL_SWEEP_INTERVAL`` constant in
:mod:`membrane.constants` that was declared but never wired.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tombstone:
    """Marker recording that a fragment was soft-deleted.

    Attributes:
        content_hash: Hash of the now-deleted fragment.
        until: Wall-clock time after which the tombstone may be
            treated as expired and the directory entry purged.
        nodes: Set of node identifiers that announced the
            deletion. Optional; empty is fine for local-only
            deletes.
    """

    content_hash: str
    until: float
    nodes: frozenset[str] = field(default_factory=frozenset)


class RefCount:
    """In-process reference counter for content hashes.

    The counter maps ``content_hash -> set[node_id]`` of the
    nodes that have a live copy. ``release`` returns ``True``
    when the last reference is gone, so the caller knows it may
    safely delete the canonical bytes.

    Thread safety:
        All public methods are guarded by an internal
        :class:`threading.RLock`. Safe to share across threads.
    """

    def __init__(self) -> None:
        """Initialize an empty counter."""
        self._refs: dict[str, set[str]] = {}
        self.lock = threading.RLock()

    def acquire(self, content_hash: str, node_id: str) -> None:
        """Record that ``node_id`` now holds ``content_hash``.

        Args:
            content_hash: Hash to register.
            node_id: Node identifier of the new holder.
        """
        with self.lock:
            self._refs.setdefault(content_hash, set()).add(node_id)

    def release(self, content_hash: str, node_id: str) -> bool:
        """Release one reference for ``content_hash``.

        Args:
            content_hash: Hash to drop.
            node_id: Holder of the reference being released.

        Returns:
            bool: ``True`` when this call dropped the last
            reference; ``False`` while at least one reference
            remains.
        """
        with self.lock:
            holders = self._refs.get(content_hash)
            if holders is None:
                return True
            holders.discard(node_id)
            if not holders:
                self._refs.pop(content_hash, None)
                return True
            return False

    def holders(self, content_hash: str) -> set[str]:
        """Return a copy of the current holder set."""
        with self.lock:
            return set(self._refs.get(content_hash, set()))

    def is_active(self, content_hash: str) -> bool:
        """Return whether ``content_hash`` has any holder."""
        with self.lock:
            return content_hash in self._refs

    def forget(self, content_hash: str) -> None:
        """Drop every reference for ``content_hash`` without erasing.

        Used after a hard delete so the counter doesn't grow
        unboundedly. Idempotent.

        Args:
            content_hash: Hash to scrub.
        """
        with self.lock:
            self._refs.pop(content_hash, None)

    def total(self) -> int:
        """Return the number of distinct hashes currently held."""
        with self.lock:
            return len(self._refs)

    def __len__(self) -> int:
        """Return the number of distinct hashes tracked."""
        return self.total()


class TombstoneTable:
    """Thread-safe tombstone table with deadline-based expiry.

    Stores ``Tombstone`` records keyed by content hash. A
    tombstone is "active" while ``time.time() < until``. After
    that it is treated as expired and may be purged.
    """

    def __init__(self) -> None:
        """Initialize an empty table."""
        self._tombstones: dict[str, Tombstone] = {}
        self.lock = threading.RLock()

    def record(self, content_hash: str, until: float, node_ids: set[str] | None = None) -> Tombstone:
        """Record or refresh a tombstone.

        Idempotent on repeat calls for the same hash; the
        ``until`` value is updated so a later gossip delivery
        can extend the lifetime.

        Args:
            content_hash: Hash being tombstoned.
            until: Wall-clock deadline after which the tombstone
                is no longer active.
            node_ids: Optional set of nodes announcing the
                deletion; merged with any prior entry.

        Returns:
            Tombstone: The stored record.
        """
        with self.lock:
            existing = self._tombstones.get(content_hash)
            if existing is not None:
                node_ids = node_ids or set()
                merged = set(existing.nodes) | node_ids
                record = Tombstone(
                    content_hash=content_hash,
                    until=max(existing.until, until),
                    nodes=frozenset(merged),
                )
            else:
                record = Tombstone(
                    content_hash=content_hash,
                    until=until,
                    nodes=frozenset(node_ids or ()),
                )
            self._tombstones[content_hash] = record
            return record

    def get(self, content_hash: str) -> Tombstone | None:
        """Return the active tombstone or ``None`` when absent/expired."""
        with self.lock:
            record = self._tombstones.get(content_hash)
            if record is None:
                return None
            if time.time() >= record.until:
                self._tombstones.pop(content_hash, None)
                return None
            return record

    def is_active(self, content_hash: str) -> bool:
        """Return whether ``content_hash`` has an active tombstone."""
        return self.get(content_hash) is not None

    def sweep_expired(self) -> list[str]:
        """Drop every expired tombstone; return their hashes.

        Returns:
            list[str]: The hashes whose tombstones expired during
            this sweep.
        """
        now = time.time()
        expired: list[str] = []
        with self.lock:
            for h, record in list(self._tombstones.items()):
                if now >= record.until:
                    expired.append(h)
                    self._tombstones.pop(h, None)
        return expired

    def total(self) -> int:
        """Return the number of recorded tombstones (active or not)."""
        with self.lock:
            return len(self._tombstones)

    def clear(self) -> None:
        """Drop every tombstone (used by tests)."""
        with self.lock:
            self._tombstones.clear()


class _EvictCallback(Protocol):
    """Hook signature used by :class:`Sweeper` for periodic sweeps."""

    def __call__(self) -> list[str]:
        """Return the hashes evicted during this pass."""
        ...


#: Callback signature for opportunistic hooks that observe sweep results.
SweepHook = Callable[[list[str]], None]


@dataclass
class Sweeper:
    """Background daemon that runs TTL / tombstone cleanup.

    Attributes:
        interval_sec: Seconds between passes.
        stop_event: Set by :meth:`stop` to terminate the loop.
        thread: The worker thread; ``None`` until :meth:`start` runs.
        on_evict_expired: Optional callback invoked with the
            list of evicted hashes after each pass.
        on_tombstones_expired: Optional callback invoked with
            the list of expired tombstone hashes after each pass.
    """

    interval_sec: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    on_evict_expired: SweepHook | None = None
    on_tombstones_expired: SweepHook | None = None
    on_post_sweep: SweepHook | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Sanity-check the requested interval."""
        if self.interval_sec <= 0:
            raise ValueError(f"Sweeper interval_sec must be > 0, got {self.interval_sec}")

    def start(self) -> None:
        """Spawn the daemon thread. Idempotent."""
        with self._lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="membrane-sweeper",
            )
            self.thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the daemon to exit and wait briefly for the thread."""
        self.stop_event.set()
        with self._lock:
            if self.thread is not None:
                self.thread.join(timeout=timeout)
                self.thread = None

    def run_once(
        self,
        evict_expired: _EvictCallback | None = None,
        tombstones: TombstoneTable | None = None,
    ) -> None:
        """Run a single cleanup pass. Useful from tests and the
        graceful-shutdown path.

        Args:
            evict_expired: Optional callback that performs TTL
                eviction. Invoked before tombstone sweep so the
                directory is up-to-date when tombstones fire.
            tombstones: Optional :class:`TombstoneTable` whose
                expired records will be purged.
        """
        total: list[str] = []
        if evict_expired is not None:
            evicted = evict_expired()
            if self.on_evict_expired is not None and evicted:
                self.on_evict_expired(evicted)
            total.extend(evicted)
        if tombstones is not None:
            expired = tombstones.sweep_expired()
            if self.on_tombstones_expired is not None and expired:
                self.on_tombstones_expired(expired)
            total.extend(expired)
        if self.on_post_sweep is not None and total:
            # De-duplicate so the post-sweep observer sees each
            # affected hash exactly once even when both phases
            # evicted it.
            self.on_post_sweep(sorted(set(total)))

    def _run(self) -> None:
        """Worker loop. Uses ``stop_event.wait`` so ``stop`` is prompt."""
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.interval_sec):
                return


__all__ = ["RefCount", "Sweeper", "Tombstone", "TombstoneTable"]
