"""Background policies: promote hot fragments, pick a role, read through to an origin.

* :class:`Promoter` (``--promote-replicas N``) copies fragments that are
  read often and score high on reuse
  (:class:`~membrane.policy.Promotion`) to more peers, chosen by load
  (:class:`~membrane.selector.Selector`), up to ``N`` copies in total.
* :class:`RolePolicy` (``--dynamic-roles``) re-evaluates whether this
  node should act as a memory host, prefill worker, or decode worker
  (:class:`~membrane.roles.Roles`) from its own and its peers' load, and
  advertises the result in its heartbeat.
* :class:`OriginLink` (``--origin HOST:PORT``) makes this node a
  regional cache (:class:`~membrane.replica.Replica`): a read that misses
  locally is fetched from the origin, bytes verified, and kept as a
  non-primary copy.
"""

import logging
from typing import Any

from membrane.network.peer import Peer
from membrane.policy import Promotion, PromotionConfig
from membrane.replication import payload_for, replicate_fragment
from membrane.roles import NodeRole, Roles, SystemState
from membrane.selector import Selector

logger = logging.getLogger(__name__)

#: Hottest fragments considered per promotion pass.
PROMOTION_CANDIDATES = 64


class Promoter:
    """Copies hot fragments to more peers.

    Attributes:
        memory: Memory service (read counts).
        view: Cluster state.
        policy: Promotion thresholds.
        promoted: Copies made so far.
    """

    def __init__(self, memory: Any, view: Any, max_replicas: int, reuse_threshold: float = 0.7) -> None:
        """Create the promoter.

        Args:
            memory: :class:`~membrane.services.memory.MemoryService`.
            view: :class:`~membrane.services.placement.ClusterView`.
            max_replicas: Copies a hot fragment may reach (this node included).
            reuse_threshold: Reuse score a fragment needs.
        """
        self.memory = memory
        self.view = view
        self.policy = Promotion(PromotionConfig(reuse_threshold=reuse_threshold, max_replicas=max_replicas))
        self.selector = Selector()
        self.promoted = 0

    def run_once(self) -> int:
        """Promote this pass's hot fragments.

        Returns:
            int: Copies made.
        """
        hits = self.memory.take_hits()
        cluster = self.view.cluster
        if cluster is None or not hits:
            return 0
        telemetry = self.view.telemetry()
        node = self.view.node
        copies = 0
        for content_hash, count in hits.most_common(PROMOTION_CANDIDATES):
            fragment = node.fragments.get(content_hash)
            if fragment is None:
                continue
            holders = self.view.holders(content_hash)
            candidates = [n for n in telemetry if n not in holders]
            ranked = self.selector.select_top_n(candidates, telemetry, n=len(candidates)) or sorted(
                candidates, key=lambda n: telemetry[n].latency_ms
            )
            # Every candidate sees the same demand; the ranking breaks the tie.
            result = self.policy.evaluate(fragment, {peer: count for peer in ranked}, holders)
            if not result.should_promote:
                continue
            payload = payload_for(fragment, node.content_store)
            if fragment.payload_ref is not None and payload is None:
                continue
            for target in result.target_replicas:
                client = cluster.membership.get_client(target)
                if client is not None and replicate_fragment(client, fragment, payload, skip_if_present=True):
                    cluster.directory.record_fragment_location(content_hash, target)
                    copies += 1
        if copies:
            logger.info("promoted %s hot fragment copies", copies)
        self.promoted += copies
        return copies


class RolePolicy:
    """Re-evaluates the node's role from cluster load.

    Attributes:
        view: Cluster state.
        role: The current role.
        changes: Role changes so far.
    """

    def __init__(self, view: Any, initial: NodeRole = NodeRole.MEMORY_HOST) -> None:
        """Create the policy.

        Args:
            view: :class:`~membrane.services.placement.ClusterView`.
            initial: Role before the first evaluation.
        """
        self.view = view
        self.roles = Roles()
        self.role = initial
        self.changes = 0

    def run_once(self) -> NodeRole:
        """Evaluate and adopt the recommended role.

        Returns:
            NodeRole: The role now held.
        """
        telemetry = list(self.view.telemetry().values())
        count = max(1, len(telemetry))
        gpu = sum(t.gpu_load for t in telemetry) / count
        memory = sum(t.memory_pressure for t in telemetry) / count
        state = SystemState(total_compute_demand=gpu, total_memory_demand=memory, average_gpu_load=gpu)
        role = self.roles.evaluate_role(self.view.node, state)
        if role != self.role:
            logger.info("node role %s -> %s", self.role.value, role.value)
            self.role = role
            self.changes += 1
        return role


def copy_fragment(peer: Any, node: Any, content_hash: str) -> Any:
    """Copy one fragment, bytes verified, from a peer into the local node.

    The copy is never primary and keeps the fragment's tenant.

    Args:
        peer: Client for the node holding it (:class:`~membrane.network.peer.Peer`).
        node: The local node.
        content_hash: Content hash.

    Returns:
        Fragment | None: The stored fragment, or ``None`` when the peer does
        not have it or its bytes fail verification.
    """
    fragment = peer.retrieve_fragment(content_hash)
    if fragment is None:
        return None
    if fragment.payload_ref is not None:
        payload = peer.get_blob(fragment.payload_ref)  # digest-checked
        if payload is None:
            logger.warning("peer %s has no verified bytes for %s", getattr(peer, "base_url", "?"), content_hash)
            return None
        node.content_store.put(fragment.payload_ref, payload)
    return fragment if node.store(fragment, is_primary=False) else None


class OriginLink:
    """Read-through to an origin node for a regional cache.

    Attributes:
        node: The local (replica) node.
        peer: Client for the origin.
        fetched: Fragments fetched so far.
    """

    def __init__(self, node: Any, origin: str, peer: Peer | None = None) -> None:
        """Connect to ``origin``.

        Args:
            node: The local :class:`~membrane.replica.Replica`.
            origin: The origin's ``host:port`` (or URL).
            peer: Pre-built client (tests).
        """
        self.node = node
        self.origin = origin
        self.peer = peer or Peer(origin)
        self.fetched = 0

    def fetch(self, content_hash: str) -> bool:
        """Copy ``content_hash`` (bytes verified) from the origin.

        Args:
            content_hash: Content hash.

        Returns:
            bool: True when the fragment is now held locally.
        """
        stored = copy_fragment(self.peer, self.node, content_hash) is not None
        if stored:
            self.fetched += 1
        return stored


__all__ = ["PROMOTION_CANDIDATES", "OriginLink", "Promoter", "RolePolicy", "copy_fragment"]
