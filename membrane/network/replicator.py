"""Replicator loop: push missing primary shards to replica nodes.

For every primary hash held locally, ask each replica target whether
it already has the fragment. If not, push the fragment via
:class:`Peer`.

The loop is bounded by :class:`~membrane.resilience.BulkheadPolicy`
so a flooding of new primaries cannot saturate inter-node bandwidth.
"""

from __future__ import annotations

import logging
import threading

from membrane.network.config import ClusterConfig
from membrane.network.membership import Membership
from membrane.network.peer import Peer
from membrane.node import Node
from membrane.shard import Shard

logger = logging.getLogger(__name__)


class Replicator:
    """Background shard-replication loop.

    Args:
        membership: Cluster membership table.
        shard: Shard manager (replica sets per primary).
        node: Local node (source of the primary fragments).
        config: Cluster configuration.
        stop_event: Stop signal shared across all cluster loops.
        running: Mutable bool flag.
        max_concurrent: Maximum concurrent in-flight replication
            calls (defaults to unbounded for backward compat).
    """

    def __init__(
        self,
        membership: Membership,
        shard: Shard,
        node: Node,
        config: ClusterConfig,
        stop_event: threading.Event,
        running: list[bool],
        max_concurrent: int = 0,
    ) -> None:
        self.membership = membership
        self.shard = shard
        self.node = node
        self.config = config
        self.stop_event = stop_event
        self.running = running
        self.semaphore = threading.Semaphore(max_concurrent) if max_concurrent > 0 else None

    def loop(self) -> None:
        """Push every primary hash to each missing replica."""
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
        """Push a single fragment to a peer (no-op if it already has it)."""
        client = self.membership.get_client(peer_id)
        if client is None:
            return

        def do_push() -> None:
            try:
                existing = client.retrieve_fragment(content_hash)
                if existing is not None:
                    return
                frag = self.node.retrieve(content_hash)
                if frag is not None:
                    client.request_replicate(frag)
                    logger.debug("Replicated %s to %s", content_hash, peer_id)
            except Exception as exc:
                logger.debug(
                    "Replication of %s to %s failed: %s", content_hash, peer_id, exc
                )

        if self.semaphore is None:
            do_push()
        else:
            with self.semaphore:
                do_push()


__all__ = ["Replicator"]
