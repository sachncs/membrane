"""MembraneNode: in-memory fragment storage, TTL, and graph-aware eviction."""

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
    """Statistics for a MembraneNode.

    Attributes:
        memory_used_bytes: Current memory consumption.
        memory_limit_bytes: Maximum allowed memory.
        fragment_count: Number of fragments stored.
        primary_count: Number of primary-owned fragments.
    """

    memory_used_bytes: int
    memory_limit_bytes: int
    fragment_count: int
    primary_count: int


#: Small epsilon added to reuse_score in the eviction formula to avoid
#: division by zero when a fragment has reuse_score == 0.
EVICTION_REUSE_EPSILON: float = 0.01


class MembraneNode:
    """Serving plane node that holds fragments in memory.

    Supports TTL expiry, LRU eviction weighted by reuse_score, and
    graph-aware co-eviction via an owned GraphManager.

    All public methods are thread-safe via an internal ``threading.RLock``.
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
            index_system: Optional index system. Creates one if None.
            graph_manager: Optional graph manager. Creates one if None.
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

        Args:
            fragment: Fragment to store.
            is_primary: Whether this node owns the primary shard.

        Returns:
            True if stored, False if fragment exceeds max_memory_bytes.
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

        Evicts the fragment if its TTL has expired (background TTL cleanup).

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment if present and not expired, else None.
        """
        with self._lock:
            fragment = self.fragments.get(content_hash)
            if fragment is None:
                return None

            now = time.time()
            age = now - self.insertion_times.get(content_hash, now)
            if age > fragment.ttl:
                logger.debug("Evicting expired fragment %s from %s", content_hash, self.node_id)
                self.remove_fragment(content_hash)
                return None

            self.access_times[content_hash] = now
            logger.debug("Retrieved fragment %s from %s", content_hash, self.node_id)
            return fragment

    def remove_fragment(self, content_hash: str) -> Fragment:
        """Remove a fragment from internal state and return it.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            The removed fragment.
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
            Tuple of (evicted_hashes, freed_bytes).
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
        """Phase 2: evict fragments by LRU weighted by reuse_score.

        Args:
            target_bytes: Number of bytes to free.
            now: Current timestamp.
            already_evicted: Set of hashes already evicted in prior phases.

        Returns:
            Tuple of (evicted_hashes, freed_bytes).
        """
        with self._lock:
            evicted: list[str] = []
            freed = 0
            candidates = [
                (h, frag) for h, frag in self.fragments.items() if h not in already_evicted
            ]

            def eviction_score(hash_and_frag: tuple[str, Fragment]) -> float:
                """Compute eviction priority: lower score = evict first."""
                h, frag = hash_and_frag
                last_access = self.access_times.get(h, now)
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

        Args:
            target_bytes: Number of bytes to free.
            seed_hashes: Fragments evicted in earlier phases.

        Returns:
            Tuple of (evicted_hashes, freed_bytes).
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
        """Evict fragments until target_bytes are freed.

        Eviction order:
        1. Expired fragments (past TTL).
        2. LRU weighted by reuse_score (lower reuse_score = more evictable).
        3. Graph-aware co-eviction of cold neighbors.

        Args:
            target_bytes: Number of bytes to free.
            current_time: Optional timestamp for deterministic testing.

        Returns:
            List of evicted content hashes.
        """
        if target_bytes <= 0:
            return []

        with self._lock:
            now = current_time if current_time is not None else time.time()
            evicted_hashes: list[str] = []
            freed = 0

            # Phase 1: evict expired fragments
            expired_evicted, expired_freed = self.evict_expired(target_bytes, now)
            evicted_hashes.extend(expired_evicted)
            freed += expired_freed
            if freed >= target_bytes:
                return evicted_hashes

            # Phase 2: LRU weighted by reuse_score
            already_evicted = set(evicted_hashes)
            lru_evicted, lru_freed = self.evict_lru(
                target_bytes - freed, now, already_evicted
            )
            evicted_hashes.extend(lru_evicted)
            freed += lru_freed
            if freed >= target_bytes:
                return evicted_hashes

            # Phase 3: graph-aware co-eviction
            graph_evicted, graph_freed = self.evict_graph_neighbors(
                target_bytes - freed, evicted_hashes
            )
            evicted_hashes.extend(graph_evicted)
            freed += graph_freed

            return evicted_hashes

    def get_memory_usage(self) -> int:
        """Return current memory consumption in bytes."""
        with self._lock:
            return self.memory_usage

    def get_shard_hashes(self) -> set[str]:
        """Return content hashes owned as primary by this node."""
        with self._lock:
            return set(self.primary_hashes)

    def heartbeat(self) -> float:
        """Return node load score between 0.0 and 1.0.

        Returns:
            Ratio of used memory to max memory.
        """
        if self.max_memory_bytes == 0:
            return 1.0
        return min(1.0, self.get_memory_usage() / self.max_memory_bytes)

    def get_stats(self) -> NodeStats:
        """Return current node statistics."""
        with self._lock:
            return NodeStats(
                memory_used_bytes=self.memory_usage,
                memory_limit_bytes=self.max_memory_bytes,
                fragment_count=len(self.fragments),
                primary_count=len(self.primary_hashes),
            )
