"""Where to fetch, prefill, and store: the routing service behind ``POST /route``.

A client (an inference engine's connector, or a request router in front
of a fleet) asks one node where a request's KV cache should come from.
The node answers from what it knows of the cluster: which nodes hold
which fragments (its own store plus the gossiped location directory),
which nodes own a hash on the ring, and each peer's last heartbeat
(round-trip latency, memory pressure, GPU load, region).

The policy is a plugin (``--placement``, group ``membrane.placement``):

=============== =============================================================
``ring``        Owners on the consistent-hash ring (the default).
``latency``     The lowest-latency holder (:class:`~membrane.latency.Latency`).
``selector``    Composite load score (:class:`~membrane.selector.Selector`).
``economic``    Value density minus cost (:class:`~membrane.economic.Economic`).
``joint``       Compute and memory chosen together (:class:`~membrane.joint.Joint`).
=============== =============================================================

With ``--route-threshold N`` the answer also says whether to prefill on
the caller's own engine or offload to Membrane
(:class:`~membrane.model.router.Router`); the threshold adapts to load
(:class:`~membrane.model.scheduler.DualTimescaleScheduler`). Whether to
reuse cached KV at all is costed with :class:`~membrane.cost.CostModel`.
"""

import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from membrane.compute.base import Backend
from membrane.compute.hashing import token_hash
from membrane.cost import CostModel
from membrane.economic import Economic
from membrane.fragment import Fragment
from membrane.joint import Joint
from membrane.latency import Latency
from membrane.model.router import Router
from membrane.model.scheduler import DualTimescaleScheduler, SchedulerState
from membrane.selector import Selector
from membrane.telemetry import Telemetry

logger = logging.getLogger(__name__)

#: Tokens per window when matching a prompt against the cluster directory
#: (the window compute backends address fragments by).
WINDOW_TOKENS = Backend.SIMULATE_WINDOW_SIZE
#: Relative bandwidth cost of reaching a peer in another region.
CROSS_REGION_COST = 0.5
#: Recent request lengths kept for the scheduler's long-term pass.
MAX_RECENT_LENGTHS = 10_000


@dataclass(frozen=True)
class Placement:
    """Where one fragment should come from and go to.

    Attributes:
        fetch_from: Node to read the bytes from (``""``: nobody holds them).
        store_on: Node that should keep a new copy.
        prefill_on: Node that should compute missing KV.
        reason: Short explanation.
    """

    fetch_from: str
    store_on: str
    prefill_on: str
    reason: str


@dataclass
class PeerLoad:
    """A node's last reported memory pressure, for :class:`~membrane.joint.Joint`.

    Attributes:
        node_id: Node identifier.
        pressure: Memory pressure in ``[0, 1]``.
    """

    node_id: str
    pressure: float

    def heartbeat(self) -> float:
        """Return memory pressure.

        Returns:
            float: The pressure.
        """
        return self.pressure


class ClusterView:
    """What one node knows about the cluster, for placement decisions.

    Attributes:
        node: The local node.
        cluster: The cluster manager (``None`` on a single node).
        replica_count: Copies the ring keeps of each hash.
        gpu_load: Callable reporting this node's GPU load.
    """

    def __init__(self, node: Any, cluster: Any = None, replica_count: int = 2, gpu_load: Any = None) -> None:
        """Create the view.

        Args:
            node: The local :class:`~membrane.node.Node`.
            cluster: The :class:`~membrane.network.cluster.Cluster`, if any.
            replica_count: Copies per hash on the ring.
            gpu_load: Zero-argument callable returning GPU load in
                ``[0, 1]``; ``0`` when ``None``.
        """
        self.node = node
        self.cluster = cluster
        self.replica_count = replica_count
        self.gpu_load = gpu_load

    @property
    def local_id(self) -> str:
        """This node's identifier."""
        return str(self.node.node_id)

    @property
    def region(self) -> str:
        """This node's region."""
        return str(getattr(self.node.attributes, "region", "") or "")

    def telemetry(self) -> dict[str, Telemetry]:
        """Load of every healthy node, this one included.

        A peer in another region costs bandwidth (``bandwidth_cost``
        :data:`CROSS_REGION_COST`; below the selector's health threshold, so
        a remote region is dearer but still usable).

        Returns:
            dict[str, Telemetry]: ``node_id -> Telemetry``.
        """
        local = Telemetry(
            node_id=self.local_id,
            latency_ms=0.0,
            bandwidth_cost=0.0,
            gpu_load=float(self.gpu_load()) if self.gpu_load is not None else 0.0,
            memory_pressure=float(self.node.heartbeat()),
        )
        result = {self.local_id: local}
        if self.cluster is None:
            return result
        for peer in self.cluster.membership.healthy():
            cost = CROSS_REGION_COST if self.region and peer.region and peer.region != self.region else 0.0
            telemetry = peer.telemetry()
            result[peer.node_id] = Telemetry(
                node_id=peer.node_id,
                latency_ms=telemetry.latency_ms,
                bandwidth_cost=cost,
                gpu_load=telemetry.gpu_load,
                memory_pressure=telemetry.memory_pressure,
            )
        return result

    def nodes(self) -> list[str]:
        """Every healthy node, this one first.

        Returns:
            list[str]: Node identifiers.
        """
        return list(self.telemetry())

    def holders(self, content_hash: str) -> list[str]:
        """Healthy nodes known to hold ``content_hash``, this one first.

        Args:
            content_hash: Content hash.

        Returns:
            list[str]: Node identifiers.
        """
        healthy = set(self.nodes())
        found = [self.local_id] if self.node.locate(content_hash) is not None else []
        if self.cluster is not None:
            found += sorted(n for n in self.cluster.directory.locate_fragment(content_hash) if n in healthy)
        return list(dict.fromkeys(found))

    def probable_holders(self, content_hash: str) -> list[str]:
        """Known holders, else the hash's ring owners (other than this node).

        The location registry is a bounded cache: a hash it no longer
        records is found where the ring places it.

        Args:
            content_hash: Content hash.

        Returns:
            list[str]: Node identifiers.
        """
        known = self.holders(content_hash)
        if known or self.cluster is None:
            return known
        return [n for n in self.owners(content_hash) if n != self.local_id]

    def owners(self, content_hash: str) -> list[str]:
        """Healthy ring owners of ``content_hash``, primary first.

        Args:
            content_hash: Content hash.

        Returns:
            list[str]: Node identifiers (this node alone without a cluster).
        """
        if self.cluster is None:
            return [self.local_id]
        healthy = set(self.nodes())
        try:
            ring = self.cluster.hash_ring.get_nodes(content_hash, self.replica_count + 1)
        except Exception:
            return [self.local_id]
        return [n for n in ring if n in healthy] or [self.local_id]

    def url(self, node_id: str) -> str | None:
        """Base URL of ``node_id`` (``None`` for this node or an unknown one).

        Args:
            node_id: Node identifier.

        Returns:
            str | None: The URL.
        """
        if self.cluster is None or node_id == self.local_id:
            return None
        return self.cluster.membership.get_url(node_id)

    def fragment(self, content_hash: str) -> Fragment | None:
        """The local copy of ``content_hash``, if any.

        Args:
            content_hash: Content hash.

        Returns:
            Fragment | None: The fragment.
        """
        key = self.node.locate(content_hash)
        return self.node.fragments.get(key) if key is not None else None


class PlacementPolicy(Protocol):
    """Decides where one fragment comes from and goes to."""

    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        """Place ``content_hash``.

        Args:
            view: Cluster state.
            content_hash: Content hash.
            history: Recently read hashes, most recent last.

        Returns:
            Placement: The decision.
        """
        ...


def nearest(view: ClusterView, candidates: list[str]) -> str:
    """The lowest-latency candidate (this node counts as zero).

    Args:
        view: Cluster state.
        candidates: Node identifiers.

    Returns:
        str: The nearest, or ``""`` for no candidates.
    """
    if not candidates:
        return ""
    latency = Latency({node_id: t.latency_ms for node_id, t in view.telemetry().items()})
    latency.add_latency(view.local_id, 0.0)
    return min(candidates, key=latency.get_latency)


class RingPlacement:
    """Ring owners: fetch from an owner that holds the bytes, store on the primary."""

    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        """Place ``content_hash`` by ring ownership.

        Args:
            view: Cluster state.
            content_hash: Content hash.
            history: Unused.

        Returns:
            Placement: The decision.
        """
        owners = view.owners(content_hash)
        holders = view.probable_holders(content_hash)
        owning = [n for n in owners if n in holders]
        fetch = owning[0] if owning else nearest(view, holders)
        return Placement(fetch_from=fetch, store_on=owners[0], prefill_on=view.local_id, reason="ring owner")


class LatencyPlacement:
    """Lowest latency: fetch from the nearest holder, prefill here."""

    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        """Place ``content_hash`` by measured latency.

        Args:
            view: Cluster state.
            content_hash: Content hash.
            history: Unused.

        Returns:
            Placement: The decision.
        """
        return Placement(
            fetch_from=nearest(view, view.probable_holders(content_hash)),
            store_on=view.owners(content_hash)[0],
            prefill_on=view.local_id,
            reason="nearest holder",
        )


class SelectorPlacement:
    """Composite load score over latency, GPU, memory, and bandwidth."""

    def __init__(self) -> None:
        """Use the default weights."""
        self.selector = Selector()

    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        """Place ``content_hash`` on the least-loaded healthy nodes.

        Args:
            view: Cluster state.
            content_hash: Content hash.
            history: Unused.

        Returns:
            Placement: The decision.
        """
        telemetry = view.telemetry()
        holders = view.probable_holders(content_hash)
        owners = view.owners(content_hash)
        return Placement(
            fetch_from=self.selector.select(holders, telemetry) or nearest(view, holders),
            store_on=self.selector.select(owners, telemetry) or owners[0],
            prefill_on=self.selector.select(list(telemetry), telemetry) or view.local_id,
            reason="lowest load score",
        )


class EconomicPlacement:
    """Store where value density minus cost is highest."""

    def __init__(self) -> None:
        """Use the default cost weights."""
        self.economic = Economic()

    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        """Place ``content_hash`` by expected value.

        Args:
            view: Cluster state.
            content_hash: Content hash.
            history: Recently read hashes (drives the value estimate).

        Returns:
            Placement: The decision.
        """
        telemetry = view.telemetry()
        fragment = view.fragment(content_hash)
        owners = view.owners(content_hash)
        store_on = self.economic.route(fragment, list(telemetry), telemetry, history) if fragment else owners[0]
        return Placement(
            fetch_from=nearest(view, view.probable_holders(content_hash)),
            store_on=store_on or owners[0],
            prefill_on=view.local_id,
            reason="value density minus cost" if fragment else "ring owner (fragment not held here)",
        )


class JointPlacement:
    """Choose the compute node and the memory node together."""

    def __init__(self) -> None:
        """Create the optimizer."""
        self.joint = Joint()

    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        """Place compute and memory for ``content_hash``.

        Args:
            view: Cluster state.
            content_hash: Content hash.
            history: Unused.

        Returns:
            Placement: The decision.
        """
        telemetry = view.telemetry()
        loads = [PeerLoad(node_id, t.memory_pressure) for node_id, t in telemetry.items()]
        decision = self.joint.optimize(view.fragment(content_hash), loads, telemetry)
        return Placement(
            fetch_from=nearest(view, view.probable_holders(content_hash)),
            store_on=decision.memory_node_id or view.local_id,
            prefill_on=decision.compute_node_id or view.local_id,
            reason="joint compute and memory",
        )


@dataclass(frozen=True)
class RouteAnswer:
    """The ``POST /route`` answer.

    Attributes:
        policy: Placement policy used.
        placement: Decision for the request's content.
        fragments: Cached fragments covering the prompt's prefix, each with
            where to fetch it.
        matched_tokens: Leading prompt tokens cached in the cluster.
        reuse: Whether fetching the cached KV beats recomputing it.
        offload: Prefill offload decision (``None`` without a threshold).
    """

    policy: str
    placement: Placement
    fragments: list[dict[str, Any]]
    matched_tokens: int
    reuse: bool
    offload: dict[str, Any] | None


class PlacementService:
    """Answers ``POST /route`` and runs the routing scheduler.

    Attributes:
        view: Cluster state.
        policy: Placement policy.
        policy_name: Its name.
        memory: Memory service (local prefix matches, read history).
        cost: Fetch-versus-recompute cost model.
        scheduler: Adaptive offload threshold (``None``: no offload answer).
    """

    def __init__(
        self,
        view: ClusterView,
        policy: PlacementPolicy,
        policy_name: str = "ring",
        memory: Any = None,
        route_threshold: int = 0,
        cost: CostModel | None = None,
    ) -> None:
        """Create the service.

        Args:
            view: Cluster state.
            policy: Placement policy.
            policy_name: Its name, reported in answers.
            memory: :class:`~membrane.services.memory.MemoryService`, if any.
            route_threshold: Initial offload threshold in tokens; ``0``
                leaves offload decisions out.
            cost: Cost model; default hardware and 100 Gbps when ``None``.
        """
        self.view = view
        self.policy = policy
        self.policy_name = policy_name
        self.memory = memory
        self.cost = cost or CostModel()
        self.scheduler = (
            DualTimescaleScheduler(SchedulerState(threshold=route_threshold, num_pd_p=1, num_pd_d=1))
            if route_threshold > 0
            else None
        )
        self.lengths: deque[int] = deque(maxlen=MAX_RECENT_LENGTHS)
        self.__lock = threading.Lock()

    def history(self) -> list[str]:
        """Recently read hashes, most recent last.

        Returns:
            list[str]: Hashes.
        """
        return list(self.memory.recent) if self.memory is not None else []

    def route(
        self,
        content_hash: str = "",
        tokens: list[int] | None = None,
        model_id: str = "default",
        local_cached_tokens: int = 0,
        auth_context: Any = None,
    ) -> RouteAnswer:
        """Decide where a request's KV comes from and where new KV goes.

        Args:
            content_hash: A fragment to place (alternative to ``tokens``).
            tokens: The prompt.
            model_id: Model identifier.
            local_cached_tokens: Prompt tokens already cached on the
                caller's own engine.
            auth_context: Authenticated caller (limits local matches to
                its tenant).

        Returns:
            RouteAnswer: The decision.
        """
        history = self.history()
        fragments: list[dict[str, Any]] = []
        matched = 0
        if tokens:
            matched, fragments = self.cached_prefix(tokens, model_id, auth_context, history)
            key = content_hash or token_hash(tokens)
        else:
            key = content_hash
        placement = self.policy.place(self.view, key, history)
        if content_hash and not tokens:
            fragments = [self.located(content_hash, placement.fetch_from)] if placement.fetch_from else []
        reuse = self.reuse_pays(matched, fragments)
        offload = None
        if self.scheduler is not None and tokens:
            with self.__lock:
                self.lengths.append(len(tokens))
                threshold = self.scheduler.state.effective_threshold
            decision = Router(threshold=threshold).route(
                total_length=len(tokens), cached_prefix_membrane=matched, cached_prefix_pd=local_cached_tokens
            )
            offload = asdict(decision) | {"threshold": threshold}
        return RouteAnswer(
            policy=self.policy_name,
            placement=placement,
            fragments=fragments,
            matched_tokens=matched,
            reuse=reuse,
            offload=offload,
        )

    def body(self, answer: RouteAnswer) -> dict[str, Any]:
        """Serialize an answer for the HTTP response, with node URLs.

        Args:
            answer: The answer.

        Returns:
            dict[str, Any]: JSON-ready body; ``*_url`` is ``None`` for this node.
        """
        body = asdict(answer)
        for role in ("fetch_from", "store_on", "prefill_on"):
            node_id = getattr(answer.placement, role)
            body["placement"][f"{role}_url"] = self.view.url(node_id) if node_id else None
        body["node_id"] = self.view.local_id
        return body

    def located(self, content_hash: str, node_id: str) -> dict[str, Any]:
        """Describe where to fetch one fragment.

        Args:
            content_hash: Content hash.
            node_id: Node to fetch from.

        Returns:
            dict[str, Any]: ``content_hash``, ``node_id``, and ``url`` (``None``
            for this node).
        """
        return {"content_hash": content_hash, "node_id": node_id, "url": self.view.url(node_id)}

    def cached_prefix(
        self, tokens: list[int], model_id: str, auth_context: Any, history: list[str]
    ) -> tuple[int, list[dict[str, Any]]]:
        """Leading prompt tokens cached anywhere in the cluster.

        This node's index answers first (exact, tenant-scoped). Beyond it,
        the prompt is cut into backend-sized windows and each window's
        hash is looked up in the gossiped location directory.

        Args:
            tokens: The prompt.
            model_id: Model identifier.
            auth_context: Authenticated caller.
            history: Recently read hashes.

        Returns:
            tuple[int, list[dict[str, Any]]]: Matched tokens and the fragments
            covering them, each with where to fetch it.
        """
        matched, located = 0, []
        if self.memory is not None:
            lookup = self.memory.prefix_lookup(tokens, model_id, auth_context)
            matched = int(lookup["matched_tokens"])
            located = [self.located(h, self.view.local_id) for h in lookup["fragments"]]
        if matched % WINDOW_TOKENS:
            return matched, located  # the local match ends inside a window
        while matched < len(tokens):
            window = tokens[matched : matched + WINDOW_TOKENS]
            content_hash = token_hash(window)
            holders = self.view.holders(content_hash)  # only recorded locations count as cached
            if not holders:
                break
            chosen = self.policy.place(self.view, content_hash, history).fetch_from
            located.append(self.located(content_hash, chosen if chosen in holders else holders[0]))
            matched += len(window)
        return matched, located

    def reuse_pays(self, matched: int, fragments: list[dict[str, Any]]) -> bool:
        """Whether fetching the matched KV is cheaper than recomputing it.

        Args:
            matched: Matched tokens.
            fragments: Where each covering fragment is fetched from.

        Returns:
            bool: True when reuse wins (always for KV already on this node).
        """
        if matched == 0:
            return False
        remote = [f for f in fragments if f["node_id"] != self.view.local_id]
        if not remote:
            return True
        telemetry = self.view.telemetry()
        size_mib = 0.0
        latency_sec = 0.0
        for item in remote:
            fragment = self.view.fragment(item["content_hash"])
            size_mib += (fragment.payload_size if fragment else WINDOW_TOKENS * 64) / (1 << 20)
            peer = telemetry.get(item["node_id"])
            latency_sec = max(latency_sec, (peer.latency_ms if peer else 0.0) / 1000.0)
        transfer = latency_sec + self.cost.find_cost(size_mib)
        return self.cost.reuse_is_cheaper(matched, size_mib, retrieval_latency_seconds=transfer)

    def adjust(self, queue_depth: int, max_queue_depth: int) -> int | None:
        """Short-term pass: raise the offload threshold under load, relax it after.

        Args:
            queue_depth: Requests waiting for a slot.
            max_queue_depth: Depth considered saturated.

        Returns:
            int | None: The effective threshold (``None`` without a scheduler).
        """
        if self.scheduler is None:
            return None
        with self.__lock:
            utilization = queue_depth / max_queue_depth if max_queue_depth > 0 else 0.0
            self.scheduler.state.monitor.record(min(1.0, utilization))
            return self.scheduler.short_term_adjust(queue_depth, max_queue_depth)

    def reoptimize(self) -> int | None:
        """Long-term pass: re-fit the threshold to recent request lengths.

        Returns:
            int | None: The new base threshold (``None`` without a scheduler
            or with too few observations).
        """
        if self.scheduler is None:
            return None
        with self.__lock:
            lengths = list(self.lengths)
        instances = max(2, len(self.view.nodes()))
        if len(lengths) < 100:
            return None
        self.scheduler.long_term_reoptimize(lengths, instances)
        logger.info("routing threshold re-fitted to %s tokens", self.scheduler.state.threshold)
        return self.scheduler.state.threshold


__all__ = [
    "CROSS_REGION_COST",
    "MAX_RECENT_LENGTHS",
    "WINDOW_TOKENS",
    "ClusterView",
    "EconomicPlacement",
    "JointPlacement",
    "LatencyPlacement",
    "PeerLoad",
    "Placement",
    "PlacementPolicy",
    "PlacementService",
    "RingPlacement",
    "RouteAnswer",
    "SelectorPlacement",
    "nearest",
]
