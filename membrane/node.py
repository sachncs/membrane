"""Node: in-memory fragment storage, TTL, and graph-aware eviction.

This module defines :class:`Node` and :class:`NodeAttributes`.

The :class:`NodeAttributes` dataclass carries the locality metadata
used by Phase 6: ``region`` (e.g. ``"eu-west-1"``) and ``zone``
(e.g. ``"us-east-1a"``) place a node in the deployment topology,
and ``bandwidth_class`` is a coarse-grained signal (0 = unmetered,
higher = metered) that :class:`~membrane.replicator.Replicator`
consults when picking which replica to fill. The attributes are
attached to the :class:`Node` instance and shipped in the
heartbeat response so peers can compute locality-aware
placements.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from membrane.fragment import Fragment
from membrane.graph import Graph
from membrane.index import Index
from membrane.security.tenant import (
    TenantAuthorizer,
)

if TYPE_CHECKING:
    from membrane.content_store import ContentStore


@dataclass(frozen=True)
class NodeAttributes:
    """Locality / bandwidth metadata advertised by a node.

    Attributes:
        region: Coarse deployment region, e.g. ``"us-east-1"``,
            ``"eu-west-1"``. ``"default"`` when unset.
        zone: Finer-grained availability zone, e.g. ``"us-east-1a"``;
            used by :class:`~membrane.shard.Shard`'s locality
            scoring. ``"default"`` when unset.
        bandwidth_class: Coarse metric of egress cost. ``0`` is
            unmetered (same-zone, free); higher values are
            proportional to inter-region cost. Default ``0``.
    """

    region: str = "default"
    zone: str = "default"
    bandwidth_class: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dict (used by heartbeats)."""
        return {
            "region": self.region,
            "zone": self.zone,
            "bandwidth_class": self.bandwidth_class,
        }


@dataclass(frozen=True)
class Stats:
    """Statistics for a :class:`Node`.

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


class Node:
    """Serving plane node that holds fragments in memory.

    Supports TTL expiry, LRU eviction weighted by ``reuse_score``,
    and graph-aware co-eviction via an owned
    :class:`Graph`.

    All public methods are thread-safe via an internal
    :class:`threading.RLock`.
    """

    def __init__(
        self,
        node_id: str,
        max_memory_bytes: int = 1 << 30,
        index_system: Index | None = None,
        graph: Graph | None = None,
        content_store: ContentStore | None = None,
        attributes: NodeAttributes | None = None,
    ) -> None:
        """Initialize the node.

        Args:
            node_id: Unique identifier for this node.
            max_memory_bytes: Memory budget in bytes.
            index_system: Optional index system. A fresh one is
                created when ``None``.
            graph: Optional fragment graph. A fresh one is created
                when ``None``.
            content_store: Optional :class:`~membrane.content_store.ContentStore`
                where canonical payload frames live. ``None`` uses
                a private in-process store; production nodes
                should pass ``FilesystemBlob(...)`` or a similar
                implementation so payloads survive process exit.
            attributes: Optional :class:`NodeAttributes` (region,
                zone, bandwidth_class). ``None`` falls back to
                the default ``("default", "default", 0)``
                triple so single-node deployments are unaffected.
        """
        self.node_id = node_id
        self.max_memory_bytes = max_memory_bytes
        self.index_system = index_system or Index()
        self.graph = graph or Graph()
        self.attributes = attributes or NodeAttributes()

        if content_store is None:
            from membrane.content_store import InProcessBytes

            self.content_store: ContentStore = InProcessBytes()
        else:
            self.content_store = content_store

        self.fragments: dict[str, Fragment] = {}
        self.primary_hashes: set[str] = set()
        self.access_times: dict[str, float] = {}
        self.insertion_times: dict[str, float] = {}
        self.memory_usage: int = 0
        self.lock = threading.RLock()
        logger.info("Initialized node %s with %s bytes", node_id, max_memory_bytes)

    def store(
        self,
        fragment: Fragment,
        is_primary: bool = True,
        caller_tenant: str = "",
        caller_scopes: frozenset[str] = frozenset(),
    ) -> bool:
        """Store a fragment in this node.

        Performs capacity-driven eviction when needed, registers
        the fragment in the index and graph systems, and updates
        access/insertion timestamps.

        The v3.0.0 release adds a tenant scope check: a caller
        without the ``admin`` scope may only write fragments to
        their own tenant (or to the system tenant when
        ``public_writable=True``, an explicit ACL knob).
        :class:`~membrane.errors.TenantScopeError` is raised on
        a cross-tenant write.

        Args:
            fragment: Fragment to store.
            is_primary: Whether this node owns the primary shard
                for the fragment.
            caller_tenant: Tenant id of the caller; empty string
                means "unauthenticated" and the check is bypassed.
            caller_scopes: Scopes granted to the caller.

        Returns:
            bool: True if the fragment is stored (or was already
            present and refreshed), False if the fragment is
            larger than ``max_memory_bytes`` or eviction could not
            free enough space.

        Raises:
            TenantScopeError: When the caller's tenant does not
                match the fragment's tenant and the caller is
                not admin.
        """
        if caller_tenant:
            authorizer = TenantAuthorizer(
                caller_tenant=caller_tenant,
                scopes=caller_scopes,
            )
            authorizer.authorize_write(fragment.tenant_id)
        if fragment.payload_size > self.max_memory_bytes:
            logger.warning(
                "Fragment %s size %s exceeds node %s limit %s",
                fragment.identity.payload_hash,
                fragment.payload_size,
                self.node_id,
                self.max_memory_bytes,
            )
            return False

        with self.lock:
            now = time.time()
            content_hash = fragment.identity.payload_hash

            if content_hash not in self.fragments:
                required = self.memory_usage + fragment.payload_size
                if required > self.max_memory_bytes:
                    # Try to make room by evicting.
                    freed = self.evict(fragment.payload_size)
                    if self.memory_usage + fragment.payload_size > self.max_memory_bytes:
                        logger.warning(
                            "Could not store %s on %s: insufficient memory after eviction",
                            content_hash,
                            self.node_id,
                        )
                        return False
                    logger.info(
                        "Evicted %s bytes to make room for %s on %s",
                        freed,
                        content_hash,
                        self.node_id,
                    )

                self.fragments[content_hash] = fragment
                self.memory_usage += fragment.payload_size
                self.insertion_times[content_hash] = now
                self.index_system.insert(fragment, {self.node_id})
                self.graph.add_node(fragment)
                # A metadata-only fragment (payload_ref is None)
                # has no body to persist; everything else must
                # already be in the configured ContentStore by the
                # time the producer (compute backend) constructs
                # the Fragment. Verify presence so callers learn
                # about mis-piped ContentStore configurations
                # rather than discovering them on retrieval.
                if fragment.payload_ref is not None and not self.content_store.has(fragment.payload_ref):
                    logger.warning(
                        "Fragment %s on %s references missing payload_ref=%s",
                        content_hash,
                        self.node_id,
                        fragment.payload_ref,
                    )
                logger.debug("Stored fragment %s on %s", content_hash, self.node_id)

            self.access_times[content_hash] = now

            if is_primary:
                self.primary_hashes.add(content_hash)

            return True

    def retrieve(
        self,
        content_hash: str,
        caller_tenant: str = "",
        caller_scopes: frozenset[str] = frozenset(),
    ) -> Fragment | None:
        """Retrieve a fragment by content hash.

        Performs opportunistic TTL cleanup: if the fragment has
        expired, it is removed before returning ``None``.

        The v3.0.0 release adds a tenant scope check: a caller
        without the ``admin`` scope may only read fragments
        from their own tenant (or from the system tenant when
        ``public_readable=True``, the default). A cross-tenant
        read returns ``None`` so the caller cannot tell a
        forbidden read from an absent fragment.

        Args:
            content_hash: Hash to look up.
            caller_tenant: Tenant id of the caller; empty string
                means "unauthenticated" and the check is bypassed.
            caller_scopes: Scopes granted to the caller.

        Returns:
            Fragment | None: The fragment if present and the
            caller is authorized; ``None`` when the fragment is
            absent, expired, or the caller is not authorized.
        """
        with self.lock:
            fragment = self.fragments.get(content_hash)
            if fragment is None:
                return None
            if caller_tenant:
                authorizer = TenantAuthorizer(
                    caller_tenant=caller_tenant,
                    scopes=caller_scopes,
                )
                try:
                    authorizer.authorize_read(fragment.tenant_id)
                except Exception:
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
        with self.lock:
            frag = self.fragments.pop(content_hash)
            self.memory_usage -= frag.payload_size
            self.primary_hashes.discard(content_hash)
            self.access_times.pop(content_hash, None)
            self.insertion_times.pop(content_hash, None)
            # Drop the canonical frame from the active store too.
            # A None payload_ref is metadata-only; skip cleanly.
            if frag.payload_ref is not None:
                self.content_store.delete(frag.payload_ref)
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
        with self.lock:
            evicted: list[str] = []
            freed = 0
            expired = [h for h, frag in self.fragments.items() if now - self.insertion_times.get(h, now) > frag.ttl]
            for h in expired:
                if freed >= target_bytes:
                    break
                frag = self.remove_fragment(h)
                freed += frag.payload_size
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
        with self.lock:
            evicted: list[str] = []
            freed = 0
            candidates = [(h, frag) for h, frag in self.fragments.items() if h not in already_evicted]

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
                freed += frag.payload_size
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
        :meth:`Graph.eviction_neighbors` and remove any
        neighbor that is still resident on this node.

        Args:
            target_bytes: Number of bytes to free.
            seed_hashes: Fragments evicted in earlier phases.

        Returns:
            tuple[list[str], int]: ``(evicted_hashes, freed_bytes)``.
        """
        with self.lock:
            evicted: list[str] = []
            freed = 0
            for h in list(seed_hashes):
                if freed >= target_bytes:
                    break
                neighbors = self.graph.eviction_neighbors(h)
                for neighbor_hash in neighbors:
                    if neighbor_hash not in self.fragments:
                        continue
                    if freed >= target_bytes:
                        break
                    neighbor_frag = self.remove_fragment(neighbor_hash)
                    freed += neighbor_frag.payload_size
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

        with self.lock:
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
            lru_evicted, lru_freed = self.evict_lru(target_bytes - freed, now, already_evicted)
            evicted_hashes.extend(lru_evicted)
            freed += lru_freed
            if freed >= target_bytes:
                return evicted_hashes

            # Phase 3: graph-aware co-eviction.
            graph_evicted, graph_freed = self.evict_graph_neighbors(target_bytes - freed, evicted_hashes)
            evicted_hashes.extend(graph_evicted)
            freed += graph_freed

            return evicted_hashes

    def get_memory_usage(self) -> int:
        """Return current memory consumption in bytes.

        Returns:
            int: Bytes currently occupied by stored fragments.
        """
        with self.lock:
            return self.memory_usage

    def get_shard_hashes(self) -> set[str]:
        """Return content hashes owned as primary by this node.

        Returns:
            set[str]: Defensive copy of the primary shard set.
        """
        with self.lock:
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

    def get_stats(self) -> Stats:
        """Return current node statistics.

        Returns:
            Stats: Snapshot of memory usage and fragment
            counts at call time.
        """
        with self.lock:
            return Stats(
                memory_used_bytes=self.memory_usage,
                memory_limit_bytes=self.max_memory_bytes,
                fragment_count=len(self.fragments),
                primary_count=len(self.primary_hashes),
            )
