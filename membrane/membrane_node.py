"""MembraneNode: in-memory fragment storage, TTL, and graph-aware eviction.

This module defines :class:`MembraneNode`, the in-memory serving
plane node that hosts fragments on a single machine. It owns:

* A :class:`~membrane.fragment.Fragment` dictionary keyed by
  ``content_hash``.
* An :class:`~membrane.index_system.IndexSystem` that maintains
  the four in-memory indexes for the fragments it holds.
* A :class:`~membrane.graph_manager.GraphManager` used during
  graph-aware eviction.

The node enforces a ``max_memory_bytes`` budget and supports three
eviction phases:

1. **TTL expiry** — fragments whose ``ttl`` has elapsed are
   removed first.
2. **Weighted LRU** — remaining candidates are sorted by
   ``last_access / (reuse_score + ε)``; cold fragments with low
   reuse_score are evicted first.
3. **Graph-aware co-eviction** — after phases 1 and 2, the node
   inspects the graph neighbors of the evicted fragments and
   removes cold neighbors as well, packing the storage more
   tightly around hot prefixes.

Thread safety:
    All public methods are protected by a
    :class:`threading.RLock` so the node can be safely shared
    across threads (including re-entrant calls from within
    eviction phases).
"""

import logging

logger = logging.getLogger(__name__)


import time
import threading
from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.graph_manager import GraphManager
from membrane.index_system import IndexSystem


@dataclass(frozen=True)
class NodeStats:
    """Statistics for a :class:`MembraneNode`.

    Attributes:
        memory_used_bytes: Current memory consumption in bytes.
        memory_limit_bytes: Configured maximum allowed memory.
        fragment_count: Number of fragments currently stored.
        primary_count: Number of fragments owned as the primary
            shard by this node.
    """

    memory_used_bytes: int
    memory_limit_bytes: int
    fragment_count: int
    primary_count: int


#: Small epsilon added to ``reuse_score`` in the eviction formula
#: to avoid division by zero when a fragment has ``reuse_score == 0``.
EVICTION_REUSE_EPSILON: float = 0.01


class MembraneNode:
    """Serving plane node that holds fragments in memory.

    Supports TTL expiry, LRU eviction weighted by ``reuse_score``,
    and graph-aware co-eviction via an owned
    :class:`GraphManager`.

    All public methods are thread-safe via an internal
    :class:`threading.RLock`.
    """

    def __init__(
        self,
        node_id: str,
        max_memory_bytes: int = 1 << 30,
        index_system: IndexSystem | None = None,
        graph_manager: GraphManager | None = None,
    ) -> None:
        """Initialize the node.

        Args:
            node_id: Unique identifier for this node.
            max_memory_bytes: Memory budget in bytes.
            index_system: Optional index system. A fresh one is
                created when ``None``.
            graph_manager: Optional graph manager. A fresh one is
                created when ``None``.
        """
        self.node_id = node_id
        self.max_memory_bytes = max_memory_bytes
        self.index_system = index_system or IndexSystem()
        self.graph_manager = graph_manager or GraphManager()

        self.fragments: dict[str, Fragment] = {}
        self.primary_hashes: set[str] = set()
        self.access_times: dict[str, float] = {}
        self.insertion_times: dict[str, float] = {}
        self.memory_usage: int = 0
        self._lock = threading.RLock()
        logger.info("Initialized node %s with %s bytes", node_id, max_memory_bytes)

    def store(self, fragment: Fragment, is_primary: bool = True) -> bool:
        """Store a fragment in this node.

        Performs capacity-driven eviction when needed, registers
        the fragment in the index and graph systems, and updates
        access/insertion timestamps.

        Args:
            fragment: Fragment to store.
            is_primary: Whether this node owns the primary shard
                for the fragment.

        Returns:
            bool: True if the fragment is stored (or was already
            present and refreshed), False if the fragment is
            larger than ``max_memory_bytes`` or eviction could not
            free enough space.
        """
        if fragment.size > self.max_memory_bytes:
            logger.warning(
                "Fragment %s size %s exceeds node %s limit %s",
                fragment.content_hash,
                fragment.size,
                self.node_id,
                self.max_memory_bytes,
            )
            return False

        with self._lock:
            now = time.time()

            if fragment.content_hash not in self.fragments:
                required = self.memory_usage + fragment.size
                if required > self.max_memory_bytes:
                    # Try to make room by evicting.
                    freed = self.evict(fragment.size)
                    if self.memory_usage + fragment.size > self.max_memory_bytes:
                        logger.warning(
                            "Could not store %s on %s: insufficient memory after eviction",
                            fragment.content_hash,
                            self.node_id,
                        )
                        return False
                    logger.info(
                        "Evicted %s bytes to make room for %s on %s",
                        freed,
                        fragment.content_hash,
                        self.node_id,
                    )

                self.fragments[fragment.content_hash] = fragment
                self.memory_usage += fragment.size
                self.insertion_times[fragment.content_hash] = now
                self.index_system.insert(fragment, {self.node_id})
                self.graph_manager.register(fragment)
                logger.debug(
                    "Stored fragment %s on %s", fragment.content_hash, self.node_id
                )

            self.access_times[fragment.content_hash] = now

            if is_primary:
                self.primary_hashes.add(fragment.content_hash)

            return True

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Retrieve a fragment by content hash.

        Performs opportunistic TTL cleanup: if the fragment has
        expired, it is removed before returning ``None``.

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment | None: The fragment if present and not
            expired, otherwise ``None``.
        """
        with self._lock:
            fragment = self.fragments.get(content_hash)
            if fragment is None:
                return None

            now = time.time()
            age = now - self.insertion_times.get(content_hash, now)
            if age > fragment.ttl:
                # Background TTL cleanup: remove the expired entry
                # rather than returning a stale fragment.
                logger.debug("Evicting expired fragment %s from %s", content_hash, self.node_id)
                self.remove_fragment(content_hash)
                return None

            self.access_times[content_hash] = now
            logger.debug("Retrieved fragment %s from %s", content_hash, self.node_id)
            return fragment

    def remove_fragment(self, content_hash: str) -> Fragment:
        """Remove a fragment from internal state and return it.

        Caller is responsible for ensuring the fragment is present
        (the implementation pops without guarding against
        ``KeyError``).

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            Fragment: The removed fragment.
        """
        with self._lock:
            frag = self.fragments.pop(content_hash)
            self.memory_usage -= frag.size
            self.primary_hashes.discard(content_hash)
            self.access_times.pop(content_hash, None)
            self.insertion_times.pop(content_hash, None)
            return frag

    def evict_expired(
        self,
        target_bytes: int,
        now: float,
    ) -> tuple[list[str], int]:
        """Phase 1: evict fragments whose TTL has expired.

        Args:
            target_bytes: Number of bytes to free.
            now: Current timestamp.

        Returns:
            tuple[list[str], int]: ``(evicted_hashes, freed_bytes)``.
            Stops as soon as ``freed_bytes >= target_bytes``.
        """
        with self._lock:
            evicted: list[str] = []
            freed = 0
            expired = [
                h
                for h, frag in self.fragments.items()
                if now - self.insertion_times.get(h, now) > frag.ttl
            ]
            for h in expired:
                if freed >= target_bytes:
                    break
                frag = self.remove_fragment(h)
                freed += frag.size
                evicted.append(h)
            return evicted, freed

    def evict_lru(
        self,
        target_bytes: int,
        now: float,
        already_evicted: set[str],
    ) -> tuple[list[str], int]:
        """Phase 2: evict fragments by LRU weighted by ``reuse_score``.

        Args:
            target_bytes: Number of bytes to free.
            now: Current timestamp.
            already_evicted: Set of hashes already evicted in
                prior phases; these are skipped.

        Returns:
            tuple[list[str], int]: ``(evicted_hashes, freed_bytes)``.
        """
        with self._lock:
            evicted: list[str] = []
            freed = 0
            candidates = [
                (h, frag) for h, frag in self.fragments.items() if h not in already_evicted
            ]

            def eviction_score(hash_and_frag: tuple[str, Fragment]) -> float:
                """Eviction priority (lower = evict first)."""
                h, frag = hash_and_frag
                last_access = self.access_times.get(h, now)
                # Earlier access and lower reuse_score both push
                # the score down, making the candidate evict
                # earlier. The epsilon avoids division by zero.
                return last_access / (frag.reuse_score + EVICTION_REUSE_EPSILON)

            candidates.sort(key=eviction_score)

            for h, frag in candidates:
                if freed >= target_bytes:
                    break
                self.remove_fragment(h)
                freed += frag.size
                evicted.append(h)
            return evicted, freed

    def evict_graph_neighbors(
        self,
        target_bytes: int,
        seed_hashes: list[str],
    ) -> tuple[list[str], int]:
        """Phase 3: co-evict cold graph neighbors of already-evicted fragments.

        For every seed hash evicted in earlier phases, look up its
        structural neighbors via
        :meth:`GraphManager.eviction_candidates` and remove any
        neighbor that is still resident on this node.

        Args:
            target_bytes: Number of bytes to free.
            seed_hashes: Fragments evicted in earlier phases.

        Returns:
            tuple[list[str], int]: ``(evicted_hashes, freed_bytes)``.
        """
        with self._lock:
            evicted: list[str] = []
            freed = 0
            for h in list(seed_hashes):
                if freed >= target_bytes:
                    break
                neighbors = self.graph_manager.eviction_candidates(h)
                for neighbor_hash in neighbors:
                    if neighbor_hash not in self.fragments:
                        continue
                    if freed >= target_bytes:
                        break
                    neighbor_frag = self.remove_fragment(neighbor_hash)
                    freed += neighbor_frag.size
                    evicted.append(neighbor_hash)
            return evicted, freed

    def evict(
        self,
        target_bytes: int,
        current_time: float | None = None,
    ) -> list[str]:
        """Evict fragments until ``target_bytes`` are freed.

        Runs the three eviction phases in order:

        1. **Expired** — fragments past their TTL.
        2. **Weighted LRU** — sorted by
           ``last_access / (reuse_score + ε)``.
        3. **Graph-aware co-eviction** — cold neighbors of the
           already-evicted fragments.

        Args:
            target_bytes: Number of bytes to free. Non-positive
                values are a no-op.
            current_time: Optional timestamp for deterministic
                testing. Defaults to :func:`time.time`.

        Returns:
            list[str]: All evicted content hashes, in eviction
            order. May be empty if the store is already under
            the target.
        """
        if target_bytes <= 0:
            return []

        with self._lock:
            now = current_time if current_time is not None else time.time()
            evicted_hashes: list[str] = []
            freed = 0

            # Phase 1: evict expired fragments.
            expired_evicted, expired_freed = self.evict_expired(target_bytes, now)
            evicted_hashes.extend(expired_evicted)
            freed += expired_freed
            if freed >= target_bytes:
                return evicted_hashes

            # Phase 2: LRU weighted by reuse_score.
            already_evicted = set(evicted_hashes)
            lru_evicted, lru_freed = self.evict_lru(
                target_bytes - freed, now, already_evicted
            )
            evicted_hashes.extend(lru_evicted)
            freed += lru_freed
            if freed >= target_bytes:
                return evicted_hashes

            # Phase 3: graph-aware co-eviction.
            graph_evicted, graph_freed = self.evict_graph_neighbors(
                target_bytes - freed, evicted_hashes
            )
            evicted_hashes.extend(graph_evicted)
            freed += graph_freed

            return evicted_hashes

    def get_memory_usage(self) -> int:
        """Return current memory consumption in bytes.

        Returns:
            int: Bytes currently occupied by stored fragments.
        """
        with self._lock:
            return self.memory_usage

    def get_shard_hashes(self) -> set[str]:
        """Return content hashes owned as primary by this node.

        Returns:
            set[str]: Defensive copy of the primary shard set.
        """
        with self._lock:
            return set(self.primary_hashes)

    def heartbeat(self) -> float:
        """Return node load score between 0.0 and 1.0.

        Defined as ``min(1.0, used / max)``. A node whose
        ``max_memory_bytes`` is ``0`` always reports ``1.0``
        (fully loaded) to avoid division by zero.

        Returns:
            float: Load ratio in ``[0.0, 1.0]``.
        """
        if self.max_memory_bytes == 0:
            return 1.0
        return min(1.0, self.get_memory_usage() / self.max_memory_bytes)

    def get_stats(self) -> NodeStats:
        """Return current node statistics.

        Returns:
            NodeStats: Snapshot of memory usage and fragment
            counts at call time.
        """
        with self._lock:
            return NodeStats(
                memory_used_bytes=self.memory_usage,
                memory_limit_bytes=self.max_memory_bytes,
                fragment_count=len(self.fragments),
                primary_count=len(self.primary_hashes),
            )
