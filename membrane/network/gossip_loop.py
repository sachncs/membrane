"""Gossip loop: periodically exchange state with random peers.

Each round picks up to ``gossip_fanout`` healthy peers uniformly at
random and exchanges a sampled view of fragment locations and
inventory digest. The merge uses
:meth:`GossipState.merge` which converges via
``max(version_id)`` per hash under the AP merge policy.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

from membrane.network.config import ClusterConfig
from membrane.network.gossip import GossipState, PeerEndpoint
from membrane.network.membership import Membership
from membrane.network.peer import Peer
from membrane.node import Node

logger = logging.getLogger(__name__)


class Gossip:
    """Background gossip loop and inbound event handler.

    Args:
        membership: Cluster membership table.
        node: Local node (source of fragments in the digest).
        config: Cluster configuration.
        directory: Fragment-location directory.
        stop_event: Stop signal shared across all cluster loops.
        running: Mutable bool flag.
    """

    def __init__(
        self,
        membership: Membership,
        node: Node,
        config: ClusterConfig,
        directory,
        stop_event: threading.Event,
        running: list[bool],
    ) -> None:
        self.membership = membership
        self.node = node
        self.config = config
        self.directory = directory
        self.stop_event = stop_event
        self.running = running

    def build_state(self) -> GossipState:
        """Snapshot local state into a :class:`GossipState`."""
        peers = [
            PeerEndpoint(node_id=p.node_id, host=p.host, port=p.port, healthy=p.healthy)
            for p in self.membership.snapshot()
        ]
        all_hashes = list(self.node.fragments.keys())
        sample_size = min(self.config.gossip_max_fragment_entries, len(all_hashes))
        sample_hashes = random.sample(all_hashes, sample_size) if all_hashes else []
        locations: dict[str, list[str]] = {}
        for h in sample_hashes:
            locations[h] = list(self.directory.locate_fragment(h))
        digest = {h: frag.version_id for h, frag in self.node.fragments.items()}
        return GossipState(
            node_id=self.node.node_id,
            timestamp=time.time(),
            peers=peers,
            fragment_locations=locations,
            inventory_digest=digest,
        )

    def loop(self) -> None:
        """Push our gossip state to random healthy peers on each tick."""
        while self.running[0] and not self.stop_event.is_set():
            healthy = self.membership.healthy()
            if not healthy:
                self.stop_event.wait(timeout=self.config.gossip_interval_sec)
                continue
            targets = random.sample(
                healthy, min(self.config.gossip_fanout, len(healthy))
            )
            state = self.build_state()
            for target in targets:
                if self.stop_event.is_set():
                    return
                client = self.membership.get_client(target.node_id)
                if client is None:
                    continue
                try:
                    resp = client.gossip(state.to_json())
                    if resp:
                        self.handle(resp)
                except Exception as exc:
                    logger.debug("Gossip to %s failed: %s", target.node_id, exc)
            self.stop_event.wait(timeout=self.config.gossip_interval_sec)

    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply an incoming gossip payload to local state.

        Args:
            data: Incoming gossip payload (parsed JSON).

        Returns:
            dict[str, Any]: Local gossip state for the caller.
        """
        try:
            incoming = GossipState.from_json(data)
        except Exception as exc:
            logger.warning("Failed to parse gossip state: %s", exc)
            return {}

        # Add/update peers from the incoming state.
        for ep in incoming.peers:
            if ep.node_id != self.node.node_id:
                self.membership.add(ep.node_id, ep.host, ep.port)

        # Merge fragment locations into the local directory.
        for h, nodes in incoming.fragment_locations.items():
            for nid in nodes:
                self.directory.record_fragment_location(h, nid)

        # Build the response with our state.
        return self.build_state().to_json()


__all__ = ["Gossip"]
