"""Node: in-memory fragment storage, TTL, and graph-aware eviction.

This module defines :class:`Node` and :class:`NodeAttributes`.

The :class:`NodeAttributes` dataclass carries the locality metadata
used for locality-aware placement: ``region`` (e.g. ``"eu-west-1"``) and ``zone``
(e.g. ``"us-east-1a"``) place a node in the deployment topology,
and ``bandwidth_class`` is a coarse-grained signal (0 = unmetered,
higher = metered) that :class:`~membrane.replicator.Replicator`
consults when picking which replica to fill. The attributes are
attached to the :class:`Node` instance and shipped in the
heartbeat response so peers can compute locality-aware
placements.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from membrane.decision import AdmissionPolicy, TenantQuota, TinyLFU
from membrane.fragment import DEFAULT_TENANT, KEY_SEPARATOR, Fragment, fragment_key
from membrane.graph import Graph
from membrane.index import Index
from membrane.metrics import NodeMetrics
from membrane.security.tenant import has_admin_scope
from membrane.store.eviction import EVICTION_REUSE_EPSILON, EvictionPolicy, FrequencyLRU, WeightedLRU
from membrane.store.table import FragmentTable
from membrane.store.tenant_guard import NO_SCOPES, TenantGuard
from membrane.tiers import TierPolicy, select_tier
from membrane.transfer_engine_ext import AdaptiveFragmenter

if TYPE_CHECKING:
    import threading

    from membrane.content_store import ContentStore

logger = logging.getLogger(__name__)


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
        """Return a JSON-friendly dict (used by heartbeats).

        Returns:
            dict[str, object]: A JSON-friendly dict (used by heartbeats).
        """
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


class Node:
    """Serving plane node that holds fragments in memory.

    A facade over :mod:`membrane.store` components: the
    :class:`~membrane.store.table.FragmentTable` (resident fragments and
    bookkeeping), an :class:`~membrane.store.eviction.EvictionPolicy`, a
    :class:`~membrane.store.tenant_guard.TenantGuard`, plus the index and
    co-access graph. Supports TTL expiry, policy-ordered eviction, and
    graph-aware co-eviction.

    Attributes:
        node_id: Unique identifier passed at construction time.
        max_memory_bytes: Configured memory budget.
        index_system: Public :class:`~membrane.index.Index` instance
            (a fresh one is created if the caller does not supply
            one). The :class:`~membrane.reconstructor.Reconstructor`
            consumes this directly.
        graph: Public :class:`~membrane.graph.Graph` instance; the
            fragment co-access graph used by graph-aware eviction.
        content_store: The :class:`~membrane.content_store.ContentStore`
            backing the canonical payload frames; defaults to
            :class:`~membrane.content_store.InProcessBytes`.
        fragments: Public ``dict[content_hash, Fragment]``; the live
            set of fragments stored on this node.

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
        metrics: NodeMetrics | None = None,
        admission_policy: AdmissionPolicy | None = None,
        quotas: dict[str, TenantQuota] | None = None,
        tier_policy: TierPolicy | None = None,
        fragmenter: AdaptiveFragmenter | None = None,
        eviction_strategy: TinyLFU | None = None,
        eviction_policy: EvictionPolicy | None = None,
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
            metrics: Optional :class:`NodeMetrics` instance. When
                supplied, per-tenant fragment counts are recorded
                on every successful store.
            admission_policy: Optional :class:`AdmissionPolicy`
                (cost/benefit gate).
            quotas: Map of tenant_id -> :class:`TenantQuota`. When ``None`` admission is
                unbounded.
            tier_policy: Optional :class:`TierPolicy`. ``select_tier`` is consulted when storing;
                ``None`` skips tier bookkeeping.
            fragmenter: Optional :class:`AdaptiveFragmenter`. ``None`` uses a fixed
                128-token window.
            eviction_strategy: Optional :class:`TinyLFU` sketch; when given
                (and ``eviction_policy`` is not), eviction orders victims by
                its frequency estimate (:class:`FrequencyLRU`).
            eviction_policy: Orders eviction victims; defaults to
                :class:`WeightedLRU`.
        """
        self.node_id = node_id
        self.max_memory_bytes = max_memory_bytes
        self.index_system = index_system or Index()
        self.graph = graph or Graph()
        self.attributes = attributes or NodeAttributes()
        self.metrics: NodeMetrics | None = metrics
        self.admission_policy = admission_policy
        self.quotas: dict[str, TenantQuota] = quotas or {}
        self.tier_policy = tier_policy
        self.fragmenter = fragmenter
        self.eviction_strategy = eviction_strategy
        self.eviction_policy: EvictionPolicy = eviction_policy or (
            FrequencyLRU(eviction_strategy) if eviction_strategy is not None else WeightedLRU()
        )
        self.table = FragmentTable()
        #: Where capacity evictions go instead of being lost (a
        #: :class:`~membrane.store.tiered.WarmTier`); ``None`` drops them.
        self.lower_tier: Any = None
        self.__selected_tiers: dict[str, str] = {}

        if content_store is None:
            from membrane.content_store import InProcessBytes

            self.content_store: ContentStore = InProcessBytes()
        else:
            self.content_store = content_store

        # When the configured store is the encrypted variant
        # (or any other tenant-aware store), the Node
        # threads ``tenant_id`` into the put / get calls so the
        # store can derive its per-tenant key.
        self.__store_uses_tenant = bool(hasattr(content_store, "tenant_id") or content_store is None)

        self.__eviction_callbacks: list[Callable[[object], None]] = []
        # Durability hooks (see :meth:`set_persistence_hooks`).
        self.__on_store: Callable[[Fragment, bool], None] | None = None
        self.__on_remove: Callable[[str], None] | None = None
        self.__eviction_counter: Callable[[str, int], None] | None = None
        self.share_hot_objects()
        logger.info("Initialized node %s with %s bytes", node_id, max_memory_bytes)

    def share_hot_objects(self) -> int:
        """Let every request thread read this node without refcount contention.

        On a free-threaded Python, switches the node and the objects each
        read reaches (table, its maps, content store, eviction policy, the
        clock) to
        deferred reference counting (:func:`~membrane.runtime.concurrency.share`).
        Call again after replacing one of them.

        Returns:
            int: Objects switched (0 on a GIL build).
        """
        from membrane.runtime.concurrency import share

        store_map = getattr(self.content_store, "store", None)
        return share(
            self,
            self.table,
            self.table.fragments,
            self.table.access_times,
            self.table.insertion_times,
            self.table.primary_hashes,
            self.content_store,
            store_map if isinstance(store_map, dict) else None,
            self.eviction_policy,
            self.index_system,
            time.time,
        )

    @property
    def fragments(self) -> dict[str, Fragment]:
        """Resident fragments (the table's map; iterate :meth:`fragment_snapshot`)."""
        return self.table.fragments

    @property
    def primary_hashes(self) -> set[str]:
        """Hashes this node owns as primary."""
        return self.table.primary_hashes

    @property
    def access_times(self) -> dict[str, float]:
        """Last access time per resident hash, with every thread's buffered reads merged."""
        return self.table.recency()

    @property
    def insertion_times(self) -> dict[str, float]:
        """Insertion time per resident hash."""
        return self.table.insertion_times

    @property
    def memory_usage(self) -> int:
        """Bytes held by resident fragments."""
        return self.table.memory_usage

    @memory_usage.setter
    def memory_usage(self, value: int) -> None:
        """Override the memory total (tests and restores).

        Args:
            value: New total in bytes.
        """
        self.table.memory_usage = value

    @property
    def digest(self) -> Any:
        """Bucketed inventory digest (:class:`~membrane.store.digest.InventoryDigest`)."""
        return self.table.digest

    @property
    def lock(self) -> threading.RLock:
        """The table lock; hold it to make several operations atomic."""
        return self.table.lock

    def store(
        self,
        fragment: Fragment,
        is_primary: bool = True,
        caller_tenant: str = "",
        caller_scopes: frozenset[str] = NO_SCOPES,
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
        TenantGuard.check_write(fragment.tenant_id, caller_tenant, caller_scopes)
        if self.admission_policy is not None and not self.admission_policy.should_admit(fragment.reuse_score):
            logger.debug(
                "Node %s rejected %s: reuse_score=%.3f below threshold",
                self.node_id,
                fragment.identity.payload_hash,
                fragment.reuse_score,
            )
            return False
        quota = self.quotas.get(fragment.tenant_id)
        if quota is not None and not quota.admit(fragment.payload_size):
            logger.debug(
                "Node %s rejected %s: tenant %s quota exhausted",
                self.node_id,
                fragment.identity.payload_hash,
                fragment.tenant_id,
            )
            return False
        if self.tier_policy is not None:
            self.__selected_tiers[fragment.key] = select_tier(self.tier_policy, fragment)
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
            content_hash = fragment.key

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

                self.table.add(fragment, now)
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
                self.__notify(self.__on_store, fragment, is_primary)
                if self.metrics is not None:
                    self.metrics.tenant.bump_fragment(fragment.tenant_id, 1)

            self.table.touch(content_hash, now, is_primary=is_primary)
            return True

    def retrieve(
        self,
        content_hash: str,
        caller_tenant: str = "",
        caller_scopes: frozenset[str] = NO_SCOPES,
    ) -> Fragment | None:
        """Retrieve a fragment by content hash.

        Performs opportunistic TTL cleanup: if the fragment has
        expired, it is removed before returning ``None``.

        Each tenant keeps its own copy of a content hash
        (:func:`~membrane.fragment.fragment_key`). A bare hash resolves to
        the caller's own copy first, then to the default tenant's. A caller
        allowed to read every tenant (authentication off, or ``admin``)
        falls back to any tenant's copy. A tenant-scoped key
        (``"tenant:hash"``) names one copy directly.

        The v3.0.0 release adds a tenant scope check: a caller
        without the ``admin`` scope may only read fragments
        from their own tenant (or from the system tenant when
        ``public_readable=True``, the default). A cross-tenant
        read returns ``None`` so the caller cannot tell a
        forbidden read from an absent fragment.

        Args:
            content_hash: Hash (or tenant-scoped key) to look up.
            caller_tenant: Tenant id of the caller; empty string
                means "unauthenticated" and the check is bypassed.
            caller_scopes: Scopes granted to the caller.

        Returns:
            Fragment | None: The fragment if present and the
            caller is authorized; ``None`` when the fragment is
            absent, expired, or the caller is not authorized.
        """
        if KEY_SEPARATOR in content_hash:
            return self.__retrieve_key(content_hash, caller_tenant, caller_scopes)
        if caller_tenant and caller_tenant != DEFAULT_TENANT:
            own = self.__retrieve_key(fragment_key(caller_tenant, content_hash), caller_tenant, caller_scopes)
            if own is not None:
                return own
        fragment = self.__retrieve_key(content_hash, caller_tenant, caller_scopes)
        if fragment is not None or self.table.tenant_keys.get(content_hash) is None:
            return fragment
        if caller_tenant and not has_admin_scope(caller_scopes):
            return None
        other = self.table.any_tenant_copy(content_hash)
        return self.__retrieve_key(other, caller_tenant, caller_scopes) if other is not None else None

    def locate(
        self,
        content_hash: str,
        caller_tenant: str = "",
        caller_scopes: frozenset[str] = NO_SCOPES,
    ) -> str | None:
        """The resident key :meth:`retrieve` would read for ``content_hash``.

        Resolves like :meth:`retrieve` without its side effects (no access
        recorded, no promotion, no TTL cleanup, no read check).

        Args:
            content_hash: Hash (or tenant-scoped key).
            caller_tenant: Tenant id of the caller (empty: any tenant).
            caller_scopes: Scopes granted to the caller.

        Returns:
            str | None: The storage key, or ``None`` when no visible copy is resident.
        """
        if KEY_SEPARATOR in content_hash:
            return content_hash if content_hash in self.table else None
        if caller_tenant and caller_tenant != DEFAULT_TENANT:
            own = fragment_key(caller_tenant, content_hash)
            if own in self.table:
                return own
        if content_hash in self.table:
            return content_hash
        if caller_tenant and not has_admin_scope(caller_scopes):
            return None
        return self.table.any_tenant_copy(content_hash)

    def __retrieve_key(
        self,
        content_hash: str,
        caller_tenant: str,
        caller_scopes: frozenset[str],
    ) -> Fragment | None:
        """Look up one storage key (see :meth:`retrieve`).

        Args:
            content_hash: Storage key.
            caller_tenant: Tenant id of the caller.
            caller_scopes: Scopes granted to the caller.

        Returns:
            Fragment | None: The fragment, or ``None``.
        """
        if self.lower_tier is not None and content_hash not in self.table:
            self.__promote(content_hash)
        # A hit takes no lock (single dict reads and one write), so reads
        # run in parallel on a free-threaded Python; only removing an
        # expired fragment does.
        now = time.time()
        entry = self.table.lookup(content_hash, now)
        if entry is None:
            return None
        fragment, expires = entry
        if caller_tenant and not TenantGuard.can_read(fragment.tenant_id, caller_tenant, caller_scopes):
            return None
        if now > expires:
            with self.lock:
                if self.table.is_expired(content_hash, now):
                    # Background TTL cleanup: remove the expired entry
                    # rather than returning a stale fragment.
                    logger.debug("Evicting expired fragment %s from %s", content_hash, self.node_id)
                    self.remove_fragment(content_hash)
            return None
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
            frag = self.table.pop(content_hash)
            self.__notify(self.__on_remove, content_hash)
            # Drop the canonical frame from the active store too, unless
            # another tenant's copy of the same content still uses it.
            # A None payload_ref is metadata-only; skip cleanly.
            if frag.payload_ref is not None and not self.table.references(frag.payload_ref):
                self.content_store.delete(frag.payload_ref)
            if self.metrics is not None:
                self.metrics.tenant.bump_fragment(frag.tenant_id, -1)
            return frag

    def window_size(self) -> int:
        """Return the active window size for new admissions.

        The v3.0.0 release threads the :class:`AdaptiveFragmenter`
        through the Node so callers can ask the
        node for the window it would use today. When no
        fragmenter is configured the helper returns the v2.0
        default 128.

        Returns:
            int: Positive window size in tokens.
        """
        if self.fragmenter is None:
            return 128
        return int(self.fragmenter.window_size())

    def tier_of(self, content_hash: str) -> str | None:
        """Return the tier name assigned to ``content_hash``.

        Args:
            content_hash: The fragment hash (or storage key).

        Returns:
            str | None: ``"hot"`` / ``"warm"`` / ``"cold"`` /
            ``"archival"`` when a tier policy is configured and
            the fragment was admitted; ``None`` otherwise.
        """
        return self.__selected_tiers.get(self.locate(content_hash) or content_hash)

    def record_hit(self, content_hash: str) -> None:
        """Record a cache hit on ``content_hash``.

        Args:
            content_hash: The hash (or storage key) of the accessed fragment.

        The v3.0.0 release runs the
        :class:`~membrane.decision.HitObserver` EMA on every hit
        so the cached fragment's ``reuse_score`` reflects the
        live usage distribution. When the node has a
        :class:`TinyLFU` eviction strategy the hash is also
        recorded as a sketch hit, which biases the future
        eviction decision toward keeping the key.
        """
        key = self.locate(content_hash)
        fragment = self.fragments.get(key) if key is not None else None
        if key is None or fragment is None:
            return
        self.eviction_policy.touch(key)
        # HitObserver EMA on reuse_score; the helper lives
        # at module level so we don't keep a long-lived
        # observer instance per node.
        from membrane.decision import HitObserver

        if not hasattr(self, "hit_observer"):
            self.hit_observer = HitObserver()
        self.hit_observer.record_hit(fragment)

    def evict_expired(
        self,
        target_bytes: int,
        now: float,
    ) -> tuple[list[str], list[Fragment], int]:
        """Stage 1: evict fragments whose TTL has expired.

        Args:
            target_bytes: Number of bytes to free.
            now: Current timestamp.

        Returns:
            tuple[list[str], list[Fragment], int]: ``(evicted_hashes,
            evicted_fragments, freed_bytes)``. The fragments are
            returned so the caller can fire eviction callbacks
            before the fragments are released to the GC.
        """
        with self.lock:
            evicted: list[str] = []
            evicted_fragments: list[Fragment] = []
            freed = 0
            for h in self.table.expired(now):
                if freed >= target_bytes:
                    break
                frag = self.remove_fragment(h)
                freed += frag.payload_size
                evicted.append(h)
                evicted_fragments.append(frag)
            return evicted, evicted_fragments, freed

    def evict_lru(
        self,
        target_bytes: int,
        now: float,
        already_evicted: set[str],
    ) -> tuple[list[str], list[Fragment], int]:
        """Stage 2: evict fragments in the eviction policy's order.

        Args:
            target_bytes: Number of bytes to free.
            now: Current timestamp.
            already_evicted: Set of hashes already evicted in
                prior stages; these are skipped.

        Returns:
            tuple[list[str], list[Fragment], int]: ``(evicted_hashes,
            evicted_fragments, freed_bytes)``.
        """
        with self.lock:
            evicted: list[str] = []
            evicted_fragments: list[Fragment] = []
            freed = 0
            candidates = [(h, frag) for h, frag in self.fragments.items() if h not in already_evicted]
            for h in self.eviction_policy.order(candidates, self.access_times, now):
                if freed >= target_bytes:
                    break
                self.__demote(h)
                frag = self.remove_fragment(h)
                freed += frag.payload_size
                evicted.append(h)
                evicted_fragments.append(frag)
            return evicted, evicted_fragments, freed

    def evict_graph_neighbors(
        self,
        target_bytes: int,
        seed_hashes: list[str],
    ) -> tuple[list[str], list[Fragment], int]:
        """Stage 3: co-evict cold graph neighbors of already-evicted fragments.

        For every seed hash evicted in earlier phases, look up its
        structural neighbors via
        :meth:`Graph.eviction_neighbors` and remove any
        neighbor that is still resident on this node.

        Args:
            target_bytes: Number of bytes to free.
            seed_hashes: Fragments evicted in earlier stages.

        Returns:
            tuple[list[str], list[Fragment], int]: ``(evicted_hashes,
            evicted_fragments, freed_bytes)``.
        """
        with self.lock:
            evicted: list[str] = []
            evicted_fragments: list[Fragment] = []
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
                    self.__demote(neighbor_hash)
                    neighbor_frag = self.remove_fragment(neighbor_hash)
                    freed += neighbor_frag.payload_size
                    evicted.append(neighbor_hash)
                    evicted_fragments.append(neighbor_frag)
            return evicted, evicted_fragments, freed

    def evict(
        self,
        target_bytes: int,
        current_time: float | None = None,
    ) -> list[str]:
        """Evict fragments until ``target_bytes`` are freed.

        Runs the three eviction stages in order:

        1. **Expired** — fragments past their TTL.
        2. **Policy order** — :attr:`eviction_policy` (weighted LRU by
           default: ``last_access / (reuse_score + ε)``).
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
            evicted_fragments: list[object] = []
            freed = 0

            # Stage 1: evict expired fragments.
            expired_evicted, expired_fragments, expired_freed = self.evict_expired(target_bytes, now)
            evicted_hashes.extend(expired_evicted)
            evicted_fragments.extend(expired_fragments)
            freed += expired_freed
            if freed >= target_bytes:
                self.__fire_eviction_callbacks(evicted_fragments)
                return evicted_hashes

            # Stage 2: eviction-policy order.
            already_evicted = set(evicted_hashes)
            lru_evicted, lru_fragments, lru_freed = self.evict_lru(target_bytes - freed, now, already_evicted)
            evicted_hashes.extend(lru_evicted)
            evicted_fragments.extend(lru_fragments)
            freed += lru_freed
            if freed >= target_bytes:
                self.__fire_eviction_callbacks(evicted_fragments)
                return evicted_hashes

            # Stage 3: graph-aware co-eviction.
            graph_evicted, graph_fragments, graph_freed = self.evict_graph_neighbors(
                target_bytes - freed, evicted_hashes
            )
            evicted_hashes.extend(graph_evicted)
            evicted_fragments.extend(graph_fragments)
            freed += graph_freed

            self.__fire_eviction_callbacks(evicted_fragments)
            return evicted_hashes

    def sweep_expired(self, current_time: float | None = None) -> list[str]:
        """Evict every fragment past its TTL, and nothing else.

        The periodic sweeper uses this instead of :meth:`evict`, whose
        LRU and graph phases would remove live fragments on a timer.

        Args:
            current_time: Optional timestamp for deterministic testing.

        Returns:
            list[str]: The evicted content hashes.
        """
        with self.lock:
            now = current_time if current_time is not None else time.time()
            evicted_hashes, evicted_fragments, _freed = self.evict_expired(self.memory_usage + 1, now)
            self.__fire_eviction_callbacks(list(evicted_fragments), reason="expired")
            return evicted_hashes

    def set_persistence_hooks(
        self,
        on_store: Callable[[Fragment, bool], None] | None,
        on_remove: Callable[[str], None] | None,
    ) -> None:
        """Install write-through durability hooks.

        ``on_store(fragment, is_primary)`` runs once per newly stored
        fragment and ``on_remove(content_hash)`` whenever a fragment
        leaves the node (eviction, TTL expiry, delete, or rollback).
        Hook failures are logged and never fail the node operation:
        the in-memory node stays authoritative.

        Args:
            on_store: Called with ``(fragment, is_primary)`` for each newly
                stored fragment.
            on_remove: Called with the content hash of each fragment that leaves
                the node.
        """
        with self.lock:
            self.__on_store = on_store
            self.__on_remove = on_remove

    def __notify(self, hook: Callable[..., None] | None, *args: object) -> None:
        """Call a persistence hook, logging (never raising) its failures.

        Args:
            hook: The hook to call, or ``None`` for no hook.
            *args: Positional arguments for ``method``.
        """
        if hook is None:
            return
        try:
            hook(*args)
        except Exception as exc:
            logger.warning("persistence hook failed on %s: %s", self.node_id, exc)

    def add_eviction_callback(self, callback: Callable[[object], None]) -> None:
        """Register a callback invoked on every evicted fragment.

        Args:
            callback: Callable that takes the
                :class:`membrane.fragment.Fragment` being evicted
                and returns nothing. The callback runs after the
                fragment has been removed from the in-memory
                state, so it is safe to inspect the fragment but
                not the Node.
        """
        with self.lock:
            self.__eviction_callbacks.append(callback)

    def set_eviction_counter(self, counter: Callable[[str, int], None] | None) -> None:
        """Install ``counter(reason, count)``, called after each eviction batch.

        ``reason`` is ``"expired"`` for TTL sweeps and ``"capacity"``
        when fragments were evicted to make room.

        Args:
            counter: Called with ``(reason, count)`` after each eviction batch;
                ``None`` disables.
        """
        self.__eviction_counter = counter

    def __fire_eviction_callbacks(self, evicted_fragments: list[object], reason: str = "capacity") -> None:
        """Snapshot the fragments and run the registered callbacks.

        Args:
            evicted_fragments: Fragments just evicted.
            reason: Why they were evicted (for metrics).
        """
        if evicted_fragments:
            self.__notify(self.__eviction_counter, reason, len(evicted_fragments))
        if not self.__eviction_callbacks or not evicted_fragments:
            return
        callbacks = list(self.__eviction_callbacks)
        for callback in callbacks:
            for fragment in evicted_fragments:
                try:
                    callback(fragment)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("eviction callback raised: %s", exc)

    def get_memory_usage(self) -> int:
        """Return current memory consumption in bytes.

        Returns:
            int: Bytes currently occupied by stored fragments.
        """
        with self.lock:
            return self.memory_usage

    def fragment_snapshot(self) -> dict[str, Fragment]:
        """Return a consistent copy of the fragment table.

        Background threads (gossip, repair, inventory) must iterate this
        copy: iterating :attr:`fragments` while a store or eviction runs
        raises ``RuntimeError: dictionary changed size during iteration``.

        Returns:
            dict[str, Fragment]: ``content_hash -> fragment`` at one instant.
        """
        return self.table.snapshot()

    def __demote(self, content_hash: str) -> None:
        """Hand a fragment that is being evicted for space to the lower tier.

        Args:
            content_hash: The fragment about to be removed.
        """
        if self.lower_tier is None:
            return
        fragment = self.fragments.get(content_hash)
        if fragment is None:
            return
        payload = self.content_store.get(fragment.payload_ref) if fragment.payload_ref is not None else None
        self.lower_tier.demote_later(fragment, payload)

    def __promote(self, content_hash: str) -> None:
        """Bring a fragment back from the lower tier into memory.

        Args:
            content_hash: The fragment to promote.
        """
        promoted = self.lower_tier.promote(content_hash)
        if promoted is None:
            return
        fragment, payload = promoted
        if fragment.payload_ref is not None and payload is not None:
            self.content_store.put(fragment.payload_ref, payload)
        self.store(fragment, is_primary=False)

    def get_shard_hashes(self) -> set[str]:
        """Return content hashes owned as primary by this node.

        Returns:
            set[str]: Defensive copy of the primary shard set.
        """
        with self.lock:
            return self.table.primaries()

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


__all__ = [
    "EVICTION_REUSE_EPSILON",
    "Node",
    "NodeAttributes",
    "Stats",
]
