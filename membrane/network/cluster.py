"""Cluster: composition root for cluster membership, gossip, and replication.

Owns the daemon threads and dispatches to focused subsystem classes:

* :class:`~membrane.network.membership.Membership` — peer table
* :class:`~membrane.network.heartbeat.Heartbeat` — heartbeat loop
* :class:`~membrane.network.failure.Failure` — failure-detection loop
* :class:`~membrane.network.gossip_loop.Gossip` — gossip loop + handler
* :class:`~membrane.replicator.Replicator` — replication loop
* :func:`~membrane.network.bootstrap.bootstrap` — one-shot seed join

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

from __future__ import annotations

import logging
import threading
from typing import Any

from membrane.network.bootstrap import bootstrap
from membrane.network.config import ClusterConfig
from membrane.network.failure import Failure
from membrane.network.gossip_loop import Gossip
from membrane.network.heartbeat import Heartbeat
from membrane.network.membership import Membership, PeerInfo
from membrane.network.peer import Peer
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
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.node = node
        self.config = config
        self.hash_ring = hash_ring or Ring()
        self.shard_manager = shard_manager or Shard(self.hash_ring)
        self.directory = directory or Registry()

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
        self.migrator.transfer_fn = self._migrator_callback
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

    def stop(self) -> None:
        """Signal all background threads to exit.

        Sets ``running`` to ``False`` and ``stop_event``, then
        joins each background thread with a short timeout. Threads
        that do not terminate within the timeout remain alive
        (they are daemon threads, so they will not block process
        exit).
        """
        self.running[0] = False
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)
        logger.info("Cluster stopped")

    def join(self) -> None:
        """Block until :meth:`stop` is called."""
        self.stop_event.wait()

    def bootstrap_loop(self) -> None:
        """One-shot bootstrap wrapper for the daemon thread."""
        bootstrap(self.membership, self.config, self.node_id, self.host, self.port)

    # ------------------------------------------------------------------
    # Membership API (preserved for backward compatibility)
    # ------------------------------------------------------------------

    def add_peer(self, node_id: str, host: str, port: int) -> None:
        """Add or update a peer; delegates to :meth:`Membership.add`."""
        self.membership.add(node_id, host, port)

    def remove_peer(self, node_id: str) -> bool:
        """Remove a peer; delegates to :meth:`Membership.remove`."""
        return self.membership.remove(node_id)

    def get_peers(self) -> list[dict[str, Any]]:
        """Return a snapshot of the membership table."""
        return self.membership.to_json()

    def is_peer_healthy(self, node_id: str) -> bool:
        """Return whether a peer is currently healthy."""
        p = self.membership.find(node_id)
        return p.healthy if p else False

    def get_peer_client(self, node_id: str) -> Peer | None:
        """Return the cached HTTP client for a peer."""
        return self.membership.get_client(node_id)

    def get_peer_url(self, node_id: str) -> str | None:
        """Return the HTTP base URL for a peer."""
        return self.membership.get_url(node_id)

    # ------------------------------------------------------------------
    # Event handlers (called by HTTP server) — preserved for compat
    # ------------------------------------------------------------------

    def on_peer_join(self, node_id: str, host: str, port: int) -> dict[str, Any]:
        """Handle a ``POST /join`` request from a peer."""
        self.add_peer(node_id, host, port)
        return {"success": True, "peers": self.membership.to_json()}

    def on_peer_leave(self, node_id: str) -> None:
        """Handle a ``POST /leave`` request.

        Removes the peer from membership and triggers the configured
        :class:`Migrator` to rebalance the leaving peer's primaries.
        Migration is skipped when the migrator's transfer function is
        unset (so tests that don't exercise migration don't need to
        wire it).
        """
        self.remove_peer(node_id)
        # The leaving peer may still own primaries in the local shard
        # table; collect them so the migrator can re-home them.
        leaving_hashes = {h for h, primary in self.shard_manager.primary_map.items() if primary == node_id}
        if self.migrator is not None and leaving_hashes:
            try:
                self.migrator.migrate(leaving_hashes, node_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "migrator.migrate(%d hashes, leaving=%s) failed: %s",
                    len(leaving_hashes),
                    node_id,
                    exc,
                )

    def _migrator_callback(self, content_hash: str, leaving_peer: str) -> None:
        """Default ``Migrator.transfer_fn`` that delegates to :meth:`Shard.migrate_primary`."""
        self.shard_manager.migrate_primary(
            content_hash,
            leaving_peer,
            local_node_id=self.node_id,
            node=self.node,
        )

    def on_heartbeat(self, node_id: str) -> dict[str, Any]:
        """Handle a ``POST /heartbeat`` request."""
        self.membership.record_heartbeat(node_id)
        stats = self.node.get_stats()
        return {
            "node_id": self.node_id,
            "load": self.node.heartbeat(),
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "healthy": True,
        }

    def on_gossip(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``POST /gossip`` request; delegates to :class:`Gossip`."""
        return self.gossip.handle(data)
