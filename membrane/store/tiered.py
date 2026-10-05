"""A warm tier on disk for fragments evicted from memory.

A node's memory budget (``--max-memory``) holds its hot fragments. With
``--warm-tier-bytes N``, a fragment evicted for capacity is not lost:
unless :func:`~membrane.tiers.select_tier` rates it ``archival``, it
moves, bytes included, to an encrypted on-disk :class:`WarmTier` under
``<data-dir>/warm``. Demotion is queued and written by a background
thread, so eviction (which runs under the node lock) never waits on disk. A retrieve that misses memory finds it there and
promotes it back. The warm tier keeps at most ``N`` bytes and drops its
least recently demoted fragments beyond that. TTL expiry and explicit
deletes do not demote.
"""

import json
import logging
import queue
import threading
from collections import OrderedDict
from typing import Any

from membrane.fragment import Fragment
from membrane.serialization import from_dict, to_dict
from membrane.tiers import TierPolicy, select_tier

logger = logging.getLogger(__name__)

META_PREFIX = "frag-"
DEMOTED_TIERS = frozenset({"hot", "warm", "cold"})


class WarmTier:
    """Fragments (metadata and bytes) demoted from memory, bounded by bytes.

    Attributes:
        store: Content store holding metadata (``frag-<hash>``) and bytes.
        capacity_bytes: Payload bytes kept before the oldest are dropped.
        policy: Decides which evicted fragments are worth keeping.
        demotions: Fragments demoted so far.
        promotions: Fragments promoted back to memory so far.
    """

    def __init__(self, store: Any, capacity_bytes: int, policy: TierPolicy | None = None) -> None:
        """Open the tier and index what is already on disk.

        Args:
            store: A content store (typically an encrypted
                :class:`~membrane.content_store.FilesystemBlob`).
            capacity_bytes: Payload bytes to keep.
            policy: Tier policy; the default thresholds when ``None``.
        """
        self.store = store
        self.capacity_bytes = capacity_bytes
        self.policy = policy or TierPolicy()
        self.demotions = 0
        self.promotions = 0
        self.__index: OrderedDict[str, int] = OrderedDict()
        self.__bytes = 0
        self.__lock = threading.Lock()
        self.__pending: dict[str, tuple[Fragment, bytes | None]] = {}
        self.__queue: queue.Queue[str | None] = queue.Queue()
        self.__worker: threading.Thread | None = None
        for key in sorted(getattr(store, "keys", list)()):
            if key.startswith(META_PREFIX):
                fragment = self.__load(key.removeprefix(META_PREFIX))
                if fragment is not None:
                    self.__index[fragment.identity.payload_hash] = fragment.payload_size
                    self.__bytes += fragment.payload_size

    @property
    def size_bytes(self) -> int:
        """Payload bytes currently held."""
        return self.__bytes

    def __contains__(self, content_hash: object) -> bool:
        """Whether ``content_hash`` is in the tier (written or queued).

        Args:
            content_hash: Content hash.

        Returns:
            bool: True when held.
        """
        with self.__lock:
            return content_hash in self.__index or content_hash in self.__pending

    def demote_later(self, fragment: Fragment, payload: bytes | None) -> None:
        """Queue an evicted fragment for :meth:`demote` on the writer thread; never blocks.

        Args:
            fragment: The evicted fragment.
            payload: Its KV bytes.
        """
        with self.__lock:
            self.__pending[fragment.identity.payload_hash] = (fragment, payload)
            if self.__worker is None or not self.__worker.is_alive():
                self.__worker = threading.Thread(target=self.__drain, daemon=True, name="membrane-warm-tier")
                self.__worker.start()
        self.__queue.put(fragment.identity.payload_hash)

    def flush(self) -> None:
        """Wait until queued demotions are written."""
        self.__queue.join()

    def __drain(self) -> None:
        """Write queued demotions."""
        while True:
            content_hash = self.__queue.get()
            try:
                if content_hash is None:
                    return
                with self.__lock:
                    item = self.__pending.pop(content_hash, None)
                if item is not None:
                    try:
                        self.demote(*item)
                    except Exception:
                        logger.exception("warm tier demotion of %s failed", content_hash)
            finally:
                self.__queue.task_done()

    def demote(self, fragment: Fragment, payload: bytes | None) -> bool:
        """Keep an evicted fragment unless the policy rates it archival.

        Args:
            fragment: The evicted fragment.
            payload: Its KV bytes (``None`` for a metadata-only fragment).

        Returns:
            bool: True when the fragment was kept.
        """
        if select_tier(self.policy, fragment) not in DEMOTED_TIERS:
            return False
        if fragment.payload_ref is not None and payload is None:
            return False
        if fragment.payload_size > self.capacity_bytes:
            return False
        content_hash = fragment.identity.payload_hash
        if fragment.payload_ref is not None and payload is not None:
            self.store.put(fragment.payload_ref, payload)
        self.store.put(META_PREFIX + content_hash, json.dumps(to_dict(fragment)).encode())
        with self.__lock:
            if content_hash not in self.__index:
                self.__bytes += fragment.payload_size
            self.__index[content_hash] = fragment.payload_size
            self.__index.move_to_end(content_hash)
            self.demotions += 1
            overflow = self.__evict_overflow()
        for victim in overflow:
            self.__drop(victim)
        return True

    def promote(self, content_hash: str) -> tuple[Fragment, bytes | None] | None:
        """Remove a fragment from the tier and return it with its bytes.

        Args:
            content_hash: Content hash.

        Returns:
            tuple[Fragment, bytes | None] | None: The fragment and its bytes,
            or ``None`` when it is not here (or its bytes are gone).
        """
        with self.__lock:
            pending = self.__pending.pop(content_hash, None)
            if pending is not None:
                self.promotions += 1
                return pending
            if content_hash not in self.__index:
                return None
        fragment = self.__load(content_hash)
        payload = None
        if fragment is not None and fragment.payload_ref is not None:
            payload = self.store.get(fragment.payload_ref)
        self.__drop(content_hash)
        if fragment is None or (fragment.payload_ref is not None and payload is None):
            return None
        with self.__lock:
            self.promotions += 1
        return fragment, payload

    def __load(self, content_hash: str) -> Fragment | None:
        """Read a fragment's metadata.

        Args:
            content_hash: Content hash.

        Returns:
            Fragment | None: The fragment, or ``None`` when unreadable.
        """
        raw = self.store.get(META_PREFIX + content_hash)
        if raw is None:
            return None
        try:
            return from_dict(json.loads(raw))
        except Exception as exc:
            logger.warning("warm tier entry %s unreadable: %s", content_hash, exc)
            return None

    def __evict_overflow(self) -> list[str]:
        """Pick the oldest entries beyond capacity (caller holds the lock).

        Returns:
            list[str]: Hashes to drop.
        """
        victims: list[str] = []
        total = self.__bytes
        for content_hash, size in self.__index.items():
            if total <= self.capacity_bytes:
                break
            victims.append(content_hash)
            total -= size
        return victims

    def __drop(self, content_hash: str) -> None:
        """Delete a fragment's metadata and bytes from the tier.

        Args:
            content_hash: Content hash.
        """
        fragment = self.__load(content_hash)
        if fragment is not None and fragment.payload_ref is not None:
            self.store.delete(fragment.payload_ref)
        self.store.delete(META_PREFIX + content_hash)
        with self.__lock:
            size = self.__index.pop(content_hash, None)
            if size is not None:
                self.__bytes -= size


__all__ = ["DEMOTED_TIERS", "META_PREFIX", "WarmTier"]
