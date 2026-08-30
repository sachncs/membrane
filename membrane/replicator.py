"""Replicator: replicate fragments to target nodes or remote peers.

This module defines :class:`Replicator`, the unified fragment-replication
helper. The class has two modes selected by the constructor:

* **One-shot replication** — given a set of content hashes, push each
  one to every target node via the supplied
  :class:`~membrane.transfer.TransferService`.
* **Background shard replication** — when a :class:`Cluster` membership
  table and the local node are provided, the class also exposes a
  :meth:`loop` coroutine that pushes every primary hash to each of
  its replica peers at a configured interval.

The two modes share state (``transfer_service``, ``membership``,
``node``) so the same instance can serve both ad-hoc and scheduled
replication. The ``max_concurrent`` cap protects inter-node
bandwidth when the loop is active.

Thread safety:
    The class itself is stateless beyond its references. The
    background :meth:`loop` runs as a daemon thread; cancellation
    goes through the supplied ``stop_event``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from membrane.node import Node
from membrane.transfer import TransferService

if TYPE_CHECKING:
    from membrane.network.cluster import ClusterConfig
    from membrane.network.membership import Membership
    from membrane.shard import Shard


logger = logging.getLogger(__name__)


class Replicator:
    """Replicates fragments to targets and (optionally) drives the cluster loop.

    Attributes:
        transfer_service: Service used for fragment movement.
            Held by reference so callers can substitute a custom
            implementation (e.g., one that records transfers for
            testing).
        membership: Optional :class:`Membership` table. When
            provided, :meth:`loop` is available.
        shard: Optional :class:`Shard` describing replica sets.
        node: Optional local node (source of primaries when looping).
        config: Optional :class:`ClusterConfig` providing the loop
            interval.
    """

    def __init__(
        self,
        transfer_service: TransferService | None = None,
        membership: Membership | None = None,
        shard: Shard | None = None,
        node: Node | None = None,
        config: ClusterConfig | None = None,
        stop_event: threading.Event | None = None,
        running: list[bool] | None = None,
        max_concurrent: int = 0,
    ) -> None:
        """Initialize the replicator.

        Args:
            transfer_service: Service used for fragment movement.
                A default :class:`TransferService` is created when
                ``None``.
            membership: Cluster membership table. Enables
                :meth:`loop` when provided.
            shard: Shard manager (replica sets per primary).
            node: Local node (source of the primary fragments).
            config: Cluster configuration; supplies the loop
                interval when ``membership`` is provided.
            stop_event: Stop signal shared across all cluster
                loops.
            running: Mutable bool flag.
            max_concurrent: Maximum concurrent in-flight replication
                calls (0 = unbounded).
        """
        self.transfer_service = transfer_service or TransferService()
        self.membership = membership
        self.shard = shard
        self.node = node
        self.config = config
        self.stop_event = stop_event
        self.running = running
        self.semaphore = threading.Semaphore(max_concurrent) if max_concurrent > 0 else None

    def replicate_cluster(
        self,
        component: set[str],
        source: Node,
        targets: list[Node],
    ) -> dict[str, list[str]]:
        """Replicate all fragments in ``component`` to each target node.

        For every target, iterates over the component and attempts
        each transfer independently. Failures are silently
        skipped and do not propagate to other targets or other
        fragments.

        Args:
            component: Set of content hashes to replicate.
            source: Node holding the fragments.
            targets: Nodes to receive replicas.

        Returns:
            dict[str, list[str]]: Mapping from ``target.node_id``
            to the list of hashes that were successfully
            transferred to that target.
        """
        results: dict[str, list[str]] = {}
        for target in targets:
            transferred: list[str] = []
            for h in component:
                if self.transfer_service.transfer_fragment(source, target, h):
                    transferred.append(h)
            results[target.node_id] = transferred
        return results

    def loop(self) -> None:
        """Push every primary hash to each missing replica.

        Only available when the replicator was constructed with a
        ``membership`` table and ``node``. Otherwise raises
        :class:`RuntimeError`.

        The loop iterates the node's primary hash set, asks the shard
        manager for the replica targets of each primary, and pushes
        the fragment to each peer. It sleeps on ``stop_event`` for
        the gossip interval between sweeps so the thread can be
        cleanly shut down.
        """
        if (
            self.membership is None
            or self.shard is None
            or self.node is None
            or self.config is None
            or self.stop_event is None
            or self.running is None
        ):
            raise RuntimeError(
                "Replicator.loop() requires membership, shard, node, "
                "config, stop_event, and running; provide them in the "
                "constructor before calling loop()."
            )

        while self.running[0] and not self.stop_event.is_set():
            primary_hashes = list(self.node.get_shard_hashes())
            for h in primary_hashes:
                if self.stop_event.is_set():
                    return
                replicas = self.shard.get_replicas(h)
                for peer_id in replicas:
                    if peer_id == self.node.node_id:
                        continue
                    self.push_one(h, peer_id)
            self.stop_event.wait(timeout=self.config.gossip_interval_sec)

    def push_one(self, content_hash: str, peer_id: str) -> None:
        """Push a single fragment to a peer (no-op if it already has it).

        Args:
            content_hash: Content hash of the fragment to push.
            peer_id: Destination peer id.
        """
        membership = self.membership
        node = self.node
        if membership is None or node is None:
            return

        def do_push() -> None:
            try:
                client = membership.get_client(peer_id)
                if client is None:
                    return
                existing = client.retrieve_fragment(content_hash)
                if existing is not None:
                    return
                frag = node.retrieve(content_hash)
                if frag is not None:
                    client.request_replicate(frag)
                    logger.debug("Replicated %s to %s", content_hash, peer_id)
            except Exception as exc:
                logger.debug(
                    "Replication of %s to %s failed: %s",
                    content_hash,
                    peer_id,
                    exc,
                )

        if self.semaphore is None:
            do_push()
        else:
            with self.semaphore:
                do_push()

    def repair(self, peer_id: str) -> int:
        """Run an anti-entropy round against ``peer_id``.

        Asks the peer for its inventory digest, computes the
        symmetric difference against the local node, and
        replicates every hash the peer is missing.

        The function operates on the existing :meth:`node.get_stats`
        inventory (per :class:`~membrane.node.Node`'s ``fragments``
        dict) for the local side and on
        ``client.get_inventory()`` for the peer. Phase 5 will
        upgrade the inventory exchange to a Bloom + Merkle pair
        so this method scales sub-linearly; for now the basic
        diff already covers everything anti-entropy needs.

        Args:
            peer_id: Destination peer identifier.

        Returns:
            int: Number of fragments pushed to ``peer_id`` during
            this round. ``0`` when nothing changed (the peer is
            already in sync or the round errored).
        """
        if self.membership is None or self.node is None:
            return 0
        client = self.membership.get_client(peer_id)
        if client is None:
            return 0
        try:
            remote_resp = client.get_inventory()
        except Exception as exc:
            logger.debug("repair: get_inventory from %s failed: %s", peer_id, exc)
            return 0
        if not isinstance(remote_resp, dict):
            return 0
        remote_versions: dict[str, int] = remote_resp.get("digest", {})
        local_versions = {h: frag.version_id for h, frag in self.node.fragments.items()}
        # Pull what we are missing (peer has it, we do not).
        missing_here = [h for h, v in remote_versions.items() if h not in local_versions or local_versions[h] < v]
        # Push what the peer is missing.
        missing_there = [h for h, v in local_versions.items() if h not in remote_versions or remote_versions[h] < v]
        pushed = 0
        for h in missing_there:
            self.push_one(h, peer_id)
            pushed += 1
        if missing_here:
            logger.debug(
                "repair: peer %s has %s fragments local is missing",
                peer_id,
                len(missing_here),
            )
        return pushed

    def repair_loop(self) -> None:
        """Background anti-entropy loop. Iterates healthy peers.

        Runs once on construction (or on the first ``repair_loop``
        call) and then sleeps for ``config.repair_interval_sec``
        between passes. ``stop_event`` interrupts the sleep.

        Requires the same constructor args as :meth:`loop`.
        """
        if (
            self.membership is None
            or self.config is None
            or self.stop_event is None
            or self.running is None
        ):
            raise RuntimeError("Replicator.repair_loop requires membership, config, stop_event, running")
        while self.running[0] and not self.stop_event.is_set():
            for peer in self.membership.healthy():
                if peer.node_id == self.config.node_id:
                    continue
                self.repair(peer.node_id)
            self.stop_event.wait(timeout=self.config.repair_interval_sec)


__all__ = ["Replicator"]
