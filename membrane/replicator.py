"""Replicator: replicate fragments to target nodes or remote peers.

This module defines :class:`Replicator`, the unified fragment-replication
helper. The class has two modes selected by the constructor:

* **One-shot replication** — given a set of content hashes, push each
  one to every target node via the supplied
  :class:`~membrane.transfer.TransferService`.
* **Background shard replication** — when a :class:`Cluster` membership
  table and the local node are provided, the class also exposes a
  :meth:`loop` that keeps every primary replicated on its replica
  peers: new primaries each sweep, a digest-based full pass every
  ``repair_interval_sec``.

The two modes share state (``transfer_service``, ``membership``,
``node``) so the same instance can serve both ad-hoc and scheduled
replication. The ``max_concurrent`` cap protects inter-node
bandwidth when the loop is active.

Thread safety:
    The class itself is stateless beyond its references. The
    background :meth:`loop` runs as a daemon thread; cancellation
    goes through the supplied ``stop_event``.
"""

import logging
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from membrane.node import Node
from membrane.replication import hand_off_primary, payload_for, replicate_fragment
from membrane.ring import EmptyRingError
from membrane.transfer import TransferService

if TYPE_CHECKING:
    from membrane.network.cluster import ClusterConfig
    from membrane.network.membership import Membership
    from membrane.network.strategy import Migrator
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
        migrator: Migrator | None = None,
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
            migrator: Paces primary hand-offs during rebalancing
                (:meth:`Migrator.delay` between moves); unpaced when
                ``None``.
        """
        self.transfer_service = transfer_service or TransferService()
        self.membership = membership
        self.shard = shard
        self.node = node
        self.config = config
        self.stop_event = stop_event
        self.running = running
        self.semaphore = threading.Semaphore(max_concurrent) if max_concurrent > 0 else None
        self.migrator = migrator

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
        """Keep every local primary replicated on its replica peers.

        Only available when the replicator was constructed with a
        ``membership`` table and ``node``. Otherwise raises
        :class:`RuntimeError`.

        Work per sweep (every ``gossip_interval_sec``) is proportional
        to the write rate, not the data size: only primaries that are
        new since the last sweep, or whose push failed, are checked. A
        full pass, one inventory request per peer and a push of only
        what each peer lacks, runs every ``repair_interval_sec`` to heal
        replicas lost to restarts. An empty hash ring (all peers
        restarting) skips the sweep.

        Raises:
            RuntimeError: When the replicator lacks the cluster wiring.
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

        known: set[str] = set()
        pending: set[str] = set()
        next_full = 0.0
        members: frozenset[str] = frozenset()
        while self.running[0] and not self.stop_event.is_set():
            primaries = self.node.get_shard_hashes()
            now = time.monotonic()
            healthy = frozenset(peer.node_id for peer in self.membership.healthy())
            # A membership change moves ring ownership: run a full pass
            # (re-replicate and rebalance) right away.
            full = now >= next_full or healthy != members
            members = healthy
            candidates = primaries if full else (primaries - known) | (pending & primaries)
            targets = self.replica_targets(candidates)
            if targets is not None:
                failed: set[str] = set()
                for peer_id, hashes in targets.items():
                    if self.stop_event.is_set():
                        return
                    if full:
                        failed |= self.push_missing(peer_id, hashes)
                    else:
                        failed |= {h for h in hashes if not self.push_one(h, peer_id)}
                known, pending = primaries, failed
                if full:
                    self.rebalance(set(healthy))
                    next_full = now + self.config.repair_interval_sec
            self.stop_event.wait(timeout=self.config.gossip_interval_sec)

    def replica_targets(self, hashes: set[str]) -> dict[str, list[str]] | None:
        """Group ``hashes`` by the peers that should hold a replica.

        Args:
            hashes: Primary content hashes.

        Returns:
            dict[str, list[str]] | None: Peer id to the hashes it should
            hold; ``None`` while the hash ring is empty.
        """
        if self.shard is None or self.node is None:
            return {}
        targets: dict[str, list[str]] = {}
        for content_hash in sorted(hashes):
            try:
                replicas = self.shard.get_replicas(content_hash)
            except EmptyRingError:
                return None
            for peer_id in replicas:
                if peer_id != self.node.node_id:
                    targets.setdefault(peer_id, []).append(content_hash)
        return targets

    def push_missing(self, peer_id: str, hashes: list[str]) -> set[str]:
        """Push the fragments in ``hashes`` that ``peer_id`` lacks.

        One ``GET /inventory`` replaces a probe per fragment.

        Args:
            peer_id: Destination peer id.
            hashes: Content hashes the peer should hold.

        Returns:
            set[str]: Hashes that could not be confirmed or pushed.
        """
        if self.membership is None or self.node is None:
            return set(hashes)
        client = self.membership.get_client(peer_id)
        if client is None:
            return set(hashes)
        try:
            inventory = client.get_inventory()
        except Exception as exc:
            logger.debug("replication: inventory from %s failed: %s", peer_id, exc)
            return set(hashes)
        if not isinstance(inventory, dict):
            return set(hashes)
        held = inventory.get("digest", {})
        failed: set[str] = set()
        for content_hash in hashes:
            if content_hash in held or (self.stop_event is not None and self.stop_event.is_set()):
                continue
            fragment = self.node.retrieve(content_hash)
            if fragment is None:
                continue
            try:
                payload = payload_for(fragment, self.node.content_store)
                ok = self.__guarded(partial(replicate_fragment, client, fragment, payload))
            except Exception as exc:
                logger.debug("replication of %s to %s failed: %s", content_hash, peer_id, exc)
                ok = False
            if not ok:
                failed.add(content_hash)
        return failed

    def push_one(self, content_hash: str, peer_id: str) -> bool:
        """Push a single fragment to a peer (no-op if it already has it).

        Args:
            content_hash: Content hash of the fragment to push.
            peer_id: Destination peer id.

        Returns:
            bool: True when the peer holds the fragment afterwards, or the
            fragment is no longer local (nothing to push).
        """
        membership = self.membership
        node = self.node
        if membership is None or node is None:
            return False

        def do_push() -> bool:
            client = membership.get_client(peer_id)
            if client is None:
                return False
            if client.retrieve_fragment(content_hash) is not None:
                return True
            frag = node.retrieve(content_hash)
            if frag is None:
                return True
            ok = replicate_fragment(client, frag, payload_for(frag, node.content_store))
            logger.debug("Replicated %s to %s: %s", content_hash, peer_id, ok)
            return ok

        try:
            return self.__guarded(do_push)
        except Exception as exc:
            logger.debug("Replication of %s to %s failed: %s", content_hash, peer_id, exc)
            return False

    def hand_off(self, content_hash: str, peer_id: str) -> bool:
        """Make ``peer_id`` the verified primary owner of a local primary.

        The peer receives the bytes and metadata with ``is_primary``; its
        copy is then checked (metadata present, payload digest equal to the
        local bytes). Only after that does this node drop its primary flag
        and record the new owner in the shard table.

        Args:
            content_hash: A primary held by this node.
            peer_id: The new owner.

        Returns:
            bool: True when ownership moved; False leaves this node primary.
        """
        if self.membership is None or self.node is None:
            return False
        client = self.membership.get_client(peer_id)
        fragment = self.node.retrieve(content_hash)
        if client is None or fragment is None:
            return False
        payload = payload_for(fragment, self.node.content_store)
        try:
            moved = self.__guarded(partial(hand_off_primary, client, fragment, payload))
        except Exception as exc:
            logger.warning("hand-off of %s to %s failed: %s", content_hash, peer_id, exc)
            return False
        if not moved:
            logger.warning("hand-off of %s to %s not verified; keeping ownership", content_hash, peer_id)
            return False
        self.node.primary_hashes.discard(content_hash)
        if self.shard is not None:
            self.shard.migrate_primary(content_hash, leaving_peer=self.node.node_id, local_node_id=peer_id)
        return True

    def rebalance(self, healthy: set[str]) -> int:
        """Hand off every primary the hash ring now assigns to another healthy node.

        Args:
            healthy: Node ids of healthy peers.

        Returns:
            int: Primaries moved; the rest stay here and are retried on the
            next full pass.
        """
        moved = 0
        for content_hash, owner in self.misplaced_primaries(healthy).items():
            if self.stop_event is not None and self.stop_event.is_set():
                break
            if self.hand_off(content_hash, owner):
                moved += 1
            pause = self.migrator.delay() if self.migrator is not None else 0.0
            if pause > 0 and self.stop_event is not None and self.stop_event.wait(pause):
                break
        if moved:
            logger.info("rebalanced %s primaries to their ring owners", moved)
        return moved

    def misplaced_primaries(self, healthy: set[str]) -> dict[str, str]:
        """Find local primaries that the hash ring now assigns to a healthy peer.

        Args:
            healthy: Node ids of healthy peers.

        Returns:
            dict[str, str]: Content hash to its ring-assigned owner, for
            primaries this node should hand off. Empty while the ring is
            empty.
        """
        if self.shard is None or self.node is None:
            return {}
        moves: dict[str, str] = {}
        for content_hash in sorted(self.node.get_shard_hashes()):
            try:
                owner = self.shard.hash_ring.get_node(content_hash)
            except EmptyRingError:
                return {}
            if owner != self.node.node_id and owner in healthy:
                moves[content_hash] = owner
        return moves

    def __guarded(self, call: Callable[[], bool]) -> bool:
        """Run ``call`` under the concurrency cap, if one is set.

        Args:
            call: The push to run.

        Returns:
            bool: What ``call`` returned.
        """
        if self.semaphore is None:
            return call()
        with self.semaphore:
            return call()

    def repair(self, peer_id: str) -> int:
        """Run an anti-entropy round against ``peer_id``.

        Asks the peer for its inventory digest, computes the
        symmetric difference against the local node, and
        replicates every hash the peer is missing.

        The function operates on the existing :meth:`node.get_stats`
        inventory (per :class:`~membrane.node.Node`'s ``fragments``
        dict) for the local side and on
        ``client.get_inventory()`` for the peer: one full digest per
        round, which is linear in the peer's inventory size.

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
        local_versions = {h: frag.version_id for h, frag in self.node.fragment_snapshot().items()}
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
        if self.membership is None or self.config is None or self.stop_event is None or self.running is None:
            raise RuntimeError("Replicator.repair_loop requires membership, config, stop_event, running")
        while self.running[0] and not self.stop_event.is_set():
            for peer in self.membership.healthy():
                if peer.node_id == self.config.node_id:
                    continue
                self.repair(peer.node_id)
            self.stop_event.wait(timeout=self.config.repair_interval_sec)


__all__ = ["Replicator"]
