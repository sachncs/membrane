"""Gossip protocol: state payloads and the gossip loop daemon.

This module owns:

* :class:`PeerEndpoint` — a peer's network address and health.
* :class:`GossipState` — the full payload exchanged between
  peers during a gossip round, including membership, fragment
  location samples, inventory digest, **and the active tombstone
  set** so soft-deletes converge across replicas.
* :class:`Gossip` — the background daemon that drives gossip
  rounds and handles incoming gossip payloads. Inbound
  tombstones are recorded through the configured
  :class:`~membrane.gc.TombstoneTable`.

The state classes are pure values. The :class:`Gossip` daemon
holds the local membership table, the directory, the node
reference for building snapshots, and the shared tombstone table.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field

from membrane.gc import TombstoneTable
from membrane.network.config import ClusterConfig
from membrane.network.membership import Membership
from membrane.node import Node
from membrane.serialization import JsonDict

logger = logging.getLogger(__name__)


@dataclass
class PeerEndpoint:
    """Network endpoint of a Membrane peer.

    Attributes:
        node_id: Peer's stable identifier.
        host: Peer's hostname or IP address.
        port: Peer's listen port.
        healthy: Whether the peer is currently considered
            healthy. Defaults to ``True`` (unknown peers are
            assumed healthy until proven otherwise).
    """

    node_id: str
    host: str
    port: int
    healthy: bool = True

    def to_json(self) -> JsonDict:
        """Serialize this endpoint to a JSON-compatible dict."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
        }

    @classmethod
    def from_json(cls, data: JsonDict) -> PeerEndpoint:
        """Deserialize a peer endpoint from a JSON-compatible dict."""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            healthy=data.get("healthy", True),
        )


@dataclass
class GossipState:
    """Serializable state exchanged during gossip rounds.

    A gossip state bundles five pieces of information:

    * ``peers`` — the sender's current membership view.
    * ``fragment_locations`` — a sampled subset of the
      ``content_hash -> [node_ids]`` mapping the sender knows
      about. Sampling bounds the message size; receivers fill
      in the rest via additional rounds.
    * ``inventory_digest`` — ``content_hash -> version_id`` for
      every fragment the sender holds locally. Used by
      :class:`~membrane.delta_sync.DeltaSync` to compute missing
      or outdated entries on the receiving side.
    * ``fragment_tombstones`` — ``content_hash -> until`` for
      every active soft-delete the sender has recorded. Receivers
      stamp the same tombstone on their local
      :class:`~membrane.gc.TombstoneTable` so the cluster
      converges on a single expiry.
    * ``timestamp`` — sender's wall-clock time at emission;
      receivers can use it to discard stale states.

    Attributes:
        node_id: Sender's node identifier.
        timestamp: Unix timestamp at emission.
        peers: List of known peer endpoints.
        fragment_locations: Sampled fragment-location map.
        inventory_digest: Full inventory digest.
        fragment_tombstones: Soft-delete markers with deadlines.
    """

    node_id: str
    timestamp: float
    peers: list[PeerEndpoint] = field(default_factory=list)
    fragment_locations: dict[str, list[str]] = field(default_factory=dict)
    inventory_digest: dict[str, int] = field(default_factory=dict)
    fragment_tombstones: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        """Serialize this state to a JSON-compatible dict.

        Returns:
            JsonDict: ``node_id``, ``timestamp``, ``peers``,
            ``fragment_locations``, ``inventory_digest``, and
            ``fragment_tombstones``.
        """
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "peers": [p.to_json() for p in self.peers],
            "fragment_locations": self.fragment_locations,
            "inventory_digest": self.inventory_digest,
            "fragment_tombstones": self.fragment_tombstones,
        }

    @classmethod
    def from_json(cls, data: JsonDict) -> GossipState:
        """Deserialize a gossip state from a JSON-compatible dict.

        Args:
            data: Mapping previously produced by
                :meth:`to_json`.

        Returns:
            GossipState: Reconstructed instance.
        """
        return cls(
            node_id=data["node_id"],
            timestamp=data["timestamp"],
            peers=[PeerEndpoint.from_json(p) for p in data.get("peers", [])],
            fragment_locations=dict(data.get("fragment_locations", {})),
            inventory_digest=dict(data.get("inventory_digest", {})),
            fragment_tombstones=dict(data.get("fragment_tombstones", {})),
        )

    def merge(self, other: GossipState) -> GossipState:
        """Merge another gossip state into a new combined state.

        Peer entries are de-duplicated by ``node_id`` with a
        small heuristic that prefers the healthier endpoint when
        both are present. Fragment-location lists are unioned.
        Inventory digest entries use ``max(version_id)`` per
        fragment so a stale gossip message cannot roll a
        fresher local state backward. Tombstone entries use
        ``max(until)`` per fragment so the longer-lived
        deadline wins.

        Args:
            other: State received from another peer.

        Returns:
            GossipState: New merged state with
            ``self.node_id`` and ``max(timestamps)``.
        """
        merged_peers = {p.node_id: p for p in self.peers}
        for p in other.peers:
            if p.node_id not in merged_peers:
                merged_peers[p.node_id] = p
            else:
                # Prefer the newer state for the same peer:
                # if we believe it's down but the other side
                # believes it's up, trust the other side.
                existing = merged_peers[p.node_id]
                if not existing.healthy and p.healthy:
                    merged_peers[p.node_id] = p

        merged_locations = dict(self.fragment_locations)
        for h, nodes in other.fragment_locations.items():
            current_nodes = set(merged_locations.get(h, []))
            current_nodes.update(nodes)
            merged_locations[h] = list(current_nodes)

        # Inventory digest merge uses max(version_id) per fragment to
        # converge under the AP merge policy. An entry is overwritten
        # only when the incoming version_id is strictly higher; this
        # avoids the LWW regression bug where a stale gossip message
        # could roll a fresher local state backward.
        merged_digest: dict[str, int] = dict(self.inventory_digest)
        for h, version in other.inventory_digest.items():
            if merged_digest.get(h, 0) < version:
                merged_digest[h] = version

        # Tombstone merge uses max(until) so the latest expiry
        # always wins. A node that observed the delete later
        # typically has a tighter deadline.
        merged_tombstones: dict[str, float] = dict(self.fragment_tombstones)
        for h, until in other.fragment_tombstones.items():
            existing_until = merged_tombstones.get(h, 0.0)
            if until > existing_until:
                merged_tombstones[h] = until

        return GossipState(
            node_id=self.node_id,
            timestamp=max(self.timestamp, other.timestamp),
            peers=list(merged_peers.values()),
            fragment_locations=merged_locations,
            inventory_digest=merged_digest,
            fragment_tombstones=merged_tombstones,
        )


class Gossip:
    """Background gossip loop and inbound event handler.

    Args:
        membership: Cluster membership table.
        node: Local node (source of fragments in the digest).
        config: Cluster configuration.
        directory: Fragment-location directory.
        tombstones: Shared tombstone table; incoming records are
            stamped with ``node_id`` of the sender so a peer's
            announcement can be attributed.
        stop_event: Stop signal shared across all cluster loops.
        running: Mutable bool flag.
    """

    def __init__(
        self,
        membership: Membership,
        node: Node,
        config: ClusterConfig,
        directory,
        tombstones: TombstoneTable,
        stop_event: threading.Event,
        running: list[bool],
    ) -> None:
        self.membership = membership
        self.node = node
        self.config = config
        self.directory = directory
        self.tombstones = tombstones
        self.stop_event = stop_event
        self.running = running

    def build_state(self) -> GossipState:
        """Snapshot local state into a :class:`GossipState`.

        The snapshot includes the active tombstone set so peers
        can converge on a single expiry for each deleted
        fragment. A purged tombstone (past ``until``) is skipped
        here because the local :class:`~membrane.gc.TombstoneTable`
        has already expired it.
        """
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
        # Surface every active tombstone to peers. There is no
        # sampling here because tombstone purges depend on every
        # node knowing the deadline.
        now = time.time()
        tombstone_map: dict[str, float] = {}
        with self.tombstones.lock:
            for h, record in self.tombstones._tombstones.items():  # type: ignore[attr-defined]
                if record.until > now:
                    tombstone_map[h] = record.until
        return GossipState(
            node_id=self.node.node_id,
            timestamp=time.time(),
            peers=peers,
            fragment_locations=locations,
            inventory_digest=digest,
            fragment_tombstones=tombstone_map,
        )

    def loop(self) -> None:
        """Push our gossip state to random healthy peers on each tick."""
        while self.running[0] and not self.stop_event.is_set():
            healthy = self.membership.healthy()
            if not healthy:
                self.stop_event.wait(timeout=self.config.gossip_interval_sec)
                continue
            targets = random.sample(healthy, min(self.config.gossip_fanout, len(healthy)))
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

    def handle(self, data: JsonDict) -> JsonDict:
        """Apply an incoming gossip payload to local state.

        Args:
            data: Incoming gossip payload (parsed JSON).

        Returns:
            JsonDict: Local gossip state for the caller.
        """
        try:
            incoming = GossipState.from_json(data)
        except Exception as exc:
            logger.warning("Failed to parse gossip state: %s", exc)
            return {}

        for ep in incoming.peers:
            if ep.node_id != self.node.node_id:
                self.membership.add(ep.node_id, ep.host, ep.port)

        for h, nodes in incoming.fragment_locations.items():
            for nid in nodes:
                self.directory.record_fragment_location(h, nid)

        # Tombstone convergence: stamp the sender-identified
        # records into the local table. ``record`` uses the
        # longer-lived of the two deadlines via its merge logic.
        for h, until in incoming.fragment_tombstones.items():
            self.tombstones.record(
                content_hash=h,
                until=until,
                node_ids={incoming.node_id},
            )

        return self.build_state().to_json()


__all__ = ["Gossip", "GossipState", "PeerEndpoint"]
