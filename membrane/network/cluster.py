"""Cluster: composition root for cluster membership, gossip, and replication.

Owns the daemon threads and dispatches to focused subsystem classes:

* :class:`~membrane.network.membership.Membership` — peer table
* :class:`~membrane.network.heartbeat.Heartbeat` — heartbeat loop
* :class:`~membrane.network.failure.Failure` — failure-detection loop
* :class:`~membrane.network.gossip_loop.Gossip` — gossip loop + handler
* :class:`~membrane.replicator.Replicator` — replication loop
* :meth:`Membership.join_seeds` — one-shot seed join

Public API is preserved for backward compatibility with the previous
god-class :class:`Cluster`. New code should prefer injecting the
focused subsystem classes directly.

Threading:
    * Membership mutations are protected by the
      :class:`~membrane.network.membership.Membership` lock.
    * Background loops run as daemon threads; they are stopped by
      :meth:`stop` (which sets ``stop_event`` and joins each thread
      with a short timeout).
"""

import logging
import socket
import threading

from membrane.gc import TombstoneTable
from membrane.network.config import ClusterConfig
from membrane.network.failure import Failure
from membrane.network.gossip import Gossip
from membrane.network.heartbeat import Heartbeat
from membrane.network.membership import Membership, PeerInfo
from membrane.network.strategy import (
    EagerMigrator,
    FailureDetector,
    Migrator,
    ThresholdDetector,
)
from membrane.node import Node
from membrane.registry import Registry
from membrane.replicator import Replicator
from membrane.ring import Ring
from membrane.shard import Shard

logger = logging.getLogger(__name__)


# Re-export for callers that imported from this module.
__all__ = ["Cluster", "PeerInfo"]


class Cluster:
    """Coordinates cluster membership, gossip, and replication.

    Composition root that wires together the focused subsystem classes.
    Each subsystem is exposed as an attribute for direct access when
    needed.

    Args:
        node_id: Identifier for this node.
        host: Bind host.
        port: Listen port.
        node: Local :class:`Node`.
        config: Cluster configuration.
        directory: Optional :class:`Registry`.
        hash_ring: Optional :class:`Ring`.
        shard_manager: Optional :class:`Shard`.
        failure_detector: Pluggable failure-detection strategy.
        migrator: Pluggable migration strategy.
    """

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        node: Node,
        config: ClusterConfig,
        directory: Registry | None = None,
        hash_ring: Ring | None = None,
        shard_manager: Shard | None = None,
        failure_detector: FailureDetector | None = None,
        migrator: Migrator | None = None,
        server: object | None = None,
        tombstones: TombstoneTable | None = None,
    ) -> None:
        """Compose the membership, failure-detection, gossip, and placement subsystems.

        Args:
            node_id: Node identifier.
            host: Bind host of this node.
            port: Listen port of this node.
            node: Local node whose fragments the cluster serves.
            config: Cluster configuration.
            directory: Fragment location registry; a new one by default.
            hash_ring: Ring to use for node selection. A default empty ring is
                created when ``None``.
            shard_manager: Placement manager; built on ``hash_ring`` by default.
            failure_detector: Failure-detection strategy; threshold-based by
                default.
            migrator: Primary migration strategy; eager by default.
            server: Owning server, if any.
            tombstones: Tombstone table shared with the server; a new one by
                default.
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.node = node
        self.config = config
        self.hash_ring = hash_ring or Ring()
        self.shard_manager = shard_manager or Shard(self.hash_ring)
        # When the local node carries :class:`~membrane.node.NodeAttributes`,
        # the Shard's locality_scored_assign can score replica
        # candidates by region/bandwidth. We seed the map with the
        # local attributes; remote peers are filled in by
        # :meth:`Membership.add` as they appear in the heartbeat
        # response.
        if not shard_manager and node.attributes is not None:
            self.shard_manager.node_attributes[node.node_id] = node.attributes
        self.directory = directory or Registry()
        # TransferService is injected by Server after the Cluster
        # is constructed; default to None so tests that don't
        # care about cross-node byte motion still work.
        self.transfer_service: object | None = None
        self.tombstones = tombstones or TombstoneTable()

        # Lifecycle state — must be initialized before subsystem
        # composition so subsystem constructors can reference them.
        self.running: list[bool] = [False]
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

        # Composed subsystems.
        self.membership = Membership(
            node_id=node_id,
            ring=self.hash_ring,
            shard=self.shard_manager,
            directory=self.directory,
        )
        self.failure_detector = failure_detector or ThresholdDetector(
            failure_remove_threshold=config.failure_remove_threshold
        )
        self.migrator = migrator or EagerMigrator()
        # Wire the migrator to a transfer function that re-homes the
        # leaving peer's primaries onto the local node. The function
        # updates the shard table and the local Node's primary set
        # in a single critical section so the cluster state stays
        # consistent.
        self.migrator.transfer_fn = self.on_peer_leave_rehome
        self.heartbeat = Heartbeat(self.membership, config, self.stop_event, self.running)
        self.failure = Failure(
            self.membership,
            config,
            self.stop_event,
            self.running,
            detector=self.failure_detector,
        )
        self.gossip = Gossip(
            self.membership,
            node,
            config,
            self.directory,
            self.tombstones,
            self.stop_event,
            self.running,
        )
        self.replicator = Replicator(
            membership=self.membership,
            shard=self.shard_manager,
            node=node,
            config=config,
            stop_event=self.stop_event,
            running=self.running,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background threads.

        Launches bootstrap (one-shot), heartbeat, and
        failure-detection threads unconditionally. Gossip and
        replication threads are started only when the
        corresponding ``config.enable_*`` flag is set.
        """
        self.running[0] = True
        self.stop_event.clear()

        # Bootstrap is one-shot; we still launch it as a thread so
        # it never blocks startup. The thread exits as soon as the
        # loop body completes (or when stop_event is set).
        loops = [
            (self.bootstrap_loop, "bootstrap"),
            (self.heartbeat.loop, "heartbeat"),
            (self.failure.loop, "failure-detection"),
        ]
        if self.config.enable_gossip:
            loops.append((self.gossip.loop, "gossip"))
        if self.config.enable_replication:
            loops.append((self.replicator.loop, "replication"))

        for target, name in loops:
            t = threading.Thread(target=target, daemon=True, name=f"membrane-{name}")
            t.start()
            self.threads.append(t)

        logger.info("Cluster started with %s background threads", len(loops))

    def stop(self, deadline_sec: float = 10.0) -> bool:
        """Signal all background threads to exit.

        Sets ``running`` to ``False`` and ``stop_event``, then
        joins each background thread with a deadline. Threads
        that do not terminate within the deadline remain alive
        (they are daemon threads, so they will not block process
        exit). The function returns ``False`` when at least one
        thread is still alive after the budget so the caller can
        decide whether to escalate.

        Args:
            deadline_sec: Per-thread wall-clock budget.

        Returns:
            bool: ``True`` when every background thread joined
            within the budget; ``False`` when at least one is
            still alive.
        """
        self.running[0] = False
        self.stop_event.set()
        joined_cleanly = True
        for t in self.threads:
            t.join(timeout=deadline_sec)
            if t.is_alive():
                logger.warning("cluster thread %s did not exit within %.1fs", t.name, deadline_sec)
                joined_cleanly = False
        logger.info("Cluster stopped (cleanly=%s)", joined_cleanly)
        return joined_cleanly

    def join(self) -> None:
        """Block until :meth:`stop` is called."""
        self.stop_event.wait()

    @property
    def advertise_host(self) -> str:
        """Address peers use to reach this node.

        ``config.advertise_host`` when set; otherwise the bind host,
        unless that is a wildcard address (``0.0.0.0`` / ``::``),
        which peers cannot dial, in which case the FQDN is used.
        """
        if self.config.advertise_host:
            return self.config.advertise_host
        if self.host in ("", "0.0.0.0", "::"):
            return socket.getfqdn()
        return self.host

    def bootstrap_loop(self) -> None:
        """Join a seed peer, retrying with capped backoff until one answers.

        Seeds commonly start at the same time (StatefulSet,
        compose), so a single attempt would leave a node
        permanently isolated when its seeds are not up yet.
        """
        seeds = list(self.config.peers)
        if not seeds:
            return
        delay = 1.0
        while not self.stop_event.is_set():
            if self.membership.join_seeds(
                seeds,
                local_node_id=self.node_id,
                host=self.advertise_host,
                port=self.port,
            ):
                return
            logger.warning("No seed peer reachable yet; retrying in %.0fs", delay)
            if self.stop_event.wait(delay):
                return
            delay = min(delay * 2, 30.0)

    def on_peer_leave_rehome(self, content_hash: str, leaving_peer: str) -> None:
        """Default ``Migrator.transfer_fn`` that delegates to :meth:`Shard.migrate_primary`.

        When a :class:`~membrane.transfer.TransferService` has
        been attached to the cluster (typical during
        ``Server.__init__``), the migration also forwards the
        canonical bytes through the wire path so the leaving
        peer's replicas stay in sync.

        Args:
            content_hash: Content hash of the fragment.
            leaving_peer: Identifier of the peer being removed.
        """
        self.shard_manager.migrate_primary(
            content_hash,
            leaving_peer,
            local_node_id=self.node_id,
            node=self.node,
            transfer_service=self.transfer_service,
        )
