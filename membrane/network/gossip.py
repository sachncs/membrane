"""Gossip protocol: state payloads and the gossip loop daemon.

This module owns:

* :class:`PeerEndpoint` — a peer's network address and health.
* :class:`GossipState` — the full payload exchanged between
  peers during a gossip round, including membership, fragment
  location samples, an inventory Bloom filter + Merkle root,
  and the active tombstone set so soft-deletes converge across
  replicas.
* :class:`Gossip` — the background daemon that drives gossip
  rounds and handles incoming gossip payloads. Inbound
  tombstones are recorded through the configured
  :class:`~membrane.gc.TombstoneTable`.

The state classes are pure values. The :class:`Gossip` daemon
holds the local membership table, the directory, the node
reference for building snapshots, and the shared tombstone table.

Inventory summary: ``inventory_merkle_root`` is the root of the node's
incrementally maintained bucket digest (:mod:`membrane.store.digest`) and
``inventory_size`` its fragment count; both cost O(1) to read. Nodes hold
different shards, so their roots differ by design: anti-entropy is the
replicator's bucket-by-bucket repair, not a gossip-time diff.
``inventory_bloom`` is no longer populated (kept in the wire format for
compatibility).
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field

from membrane.errors import AuthError, NetworkError, SchemaError
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
        """Serialize this endpoint to a JSON-compatible dict.

        Returns:
            JsonDict: ``node_id``, ``host``, ``port``, and ``healthy``.
        """
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
        }

    @classmethod
    def from_json(cls, data: JsonDict) -> PeerEndpoint:
        """Deserialize a peer endpoint from a JSON-compatible dict.

        Args:
            data: Mapping previously produced by :meth:`to_json`.

        Returns:
            PeerEndpoint: The endpoint.
        """
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            healthy=data.get("healthy", True),
        )


@dataclass
class GossipState:
    """Serializable state exchanged during gossip rounds.

    A gossip state bundles six pieces of information:

    * ``peers`` — the sender's current membership view.
    * ``fragment_locations`` — a sampled subset of the
      ``content_hash -> [node_ids]`` mapping the sender knows
      about. Sampling bounds the message size; receivers fill
      in the rest via additional rounds.
    * ``inventory_bloom`` — always empty; kept so older peers
      can still parse the message (repair uses bucket digests).
    * ``inventory_merkle_root`` — 32-byte root of the
      :class:`~membrane.merkle.MerkleTree` over the sender's
      ``(content_hash, owner_node_id)`` pairs. When roots
      disagree, the receiver descends the tree to find divergent
      leaves.
    * ``inventory_size`` — leaf count of the Merkle tree, used
      as a fast pre-check (size mismatch implies divergence
      without comparing the roots).
    * ``fragment_tombstones`` — ``content_hash -> until`` for
      every active soft-delete the sender has recorded.
    * ``timestamp`` — sender's wall-clock time at emission;
      receivers can use it to discard stale states.

    Attributes:
        node_id: Sender's node identifier.
        timestamp: Unix timestamp at emission.
        peers: List of known peer endpoints.
        fragment_locations: Sampled fragment-location map.
        inventory_bloom: Serialized Bloom filter bytes.
        inventory_merkle_root: 32-byte Merkle root.
        inventory_size: Number of leaves in the Merkle tree.
        fragment_tombstones: Soft-delete markers with deadlines.
    """

    node_id: str
    timestamp: float
    peers: list[PeerEndpoint] = field(default_factory=list)
    fragment_locations: dict[str, list[str]] = field(default_factory=dict)
    inventory_bloom: bytes = b""
    inventory_merkle_root: bytes = b""
    inventory_size: int = 0
    fragment_tombstones: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        """Serialize this state to a JSON-compatible dict.

        Returns:
            JsonDict: ``node_id``, ``timestamp``, ``peers``,
            ``fragment_locations``, ``inventory_bloom`` (base64
            string for the wire), ``inventory_merkle_root``
            (hex), ``inventory_size``, and
            ``fragment_tombstones``.
        """
        import base64

        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "peers": [p.to_json() for p in self.peers],
            "fragment_locations": self.fragment_locations,
            "inventory_bloom": base64.b64encode(self.inventory_bloom).decode("ascii"),
            "inventory_merkle_root": self.inventory_merkle_root.hex(),
            "inventory_size": self.inventory_size,
            "fragment_tombstones": self.fragment_tombstones,
        }

    @classmethod
    def from_json(cls, data: JsonDict) -> GossipState:
        """Deserialize a gossip state from a JSON-compatible dict.

        Args:
            data: Mapping previously produced by
                :meth:`to_json`.

        Returns:
            GossipState: Reconstructed instance. When the wire
            payload predates the Bloom/Merkle inventory (no ``inventory_bloom`` /
            ``inventory_merkle_root`` keys) the instance is built
            with empty inventory placeholders so older clusters
            still parse; the receiving side falls back to
            ``inventory_digest`` when present.
        """
        import base64

        bloom_b64 = data.get("inventory_bloom", "")
        inventory_bloom = base64.b64decode(bloom_b64) if bloom_b64 else b""
        merkle_hex = data.get("inventory_merkle_root", "")
        inventory_merkle_root = bytes.fromhex(merkle_hex) if merkle_hex else b""
        return cls(
            node_id=data["node_id"],
            timestamp=data["timestamp"],
            peers=[PeerEndpoint.from_json(p) for p in data.get("peers", [])],
            fragment_locations=dict(data.get("fragment_locations", {})),
            inventory_bloom=inventory_bloom,
            inventory_merkle_root=inventory_merkle_root,
            inventory_size=int(data.get("inventory_size", 0)),
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

        The inventory-side fields (Bloom + Merkle root)
        are not merged field-by-field because both peers
        computed them from the same local observation. The
        combined state keeps the sender-side fields so the
        receiver can chain another :meth:`Gossip.handle` pass
        to push deltas down the tree.

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
                existing = merged_peers[p.node_id]
                if not existing.healthy and p.healthy:
                    merged_peers[p.node_id] = p

        merged_locations = dict(self.fragment_locations)
        for h, nodes in other.fragment_locations.items():
            current_nodes = set(merged_locations.get(h, []))
            current_nodes.update(nodes)
            merged_locations[h] = list(current_nodes)

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
            inventory_bloom=self.inventory_bloom or other.inventory_bloom,
            inventory_merkle_root=self.inventory_merkle_root or other.inventory_merkle_root,
            inventory_size=max(self.inventory_size, other.inventory_size),
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
        """Initialize the gossip loop.

        Args:
            membership: Cluster membership table. Enables :meth:`loop` when
                provided.
            node: Local node whose inventory is gossiped.
            config: Cluster configuration.
            directory: Fragment location registry shared with the cluster.
            tombstones: Tombstone table to gossip and merge.
            stop_event: Stop signal shared across all cluster loops.
            running: Mutable bool flag.
        """
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

        Returns:
            GossipState: Snapshot of this node's gossip state.
        """
        peers = [
            PeerEndpoint(node_id=p.node_id, host=p.host, port=p.port, healthy=p.healthy)
            for p in self.membership.snapshot()
        ]
        locations: dict[str, list[str]] = {}
        for h in self.node.digest.sample(self.config.gossip_max_fragment_entries):
            # This node holds every sampled fragment.
            locations[h] = sorted(self.directory.locate_fragment(h) | {self.node.node_id})

        bloom_bytes, merkle_root, inventory_size = self.inventory_summary()

        # Surface every active tombstone to peers. There is no
        # sampling here because tombstone purges depend on every
        # node knowing the deadline.
        now = time.time()
        tombstone_map: dict[str, float] = {}
        with self.tombstones.lock:
            for h, record in self.tombstones.tombstones.items():
                if record.until > now:
                    tombstone_map[h] = record.until
        return GossipState(
            node_id=self.node.node_id,
            timestamp=time.time(),
            peers=peers,
            fragment_locations=locations,
            inventory_bloom=bloom_bytes,
            inventory_merkle_root=merkle_root,
            inventory_size=inventory_size,
            fragment_tombstones=tombstone_map,
        )

    def inventory_summary(self) -> tuple[bytes, bytes, int]:
        """Return the (unused) Bloom bytes, the inventory root, and its size.

        Both come from the node's incrementally maintained digest, so this
        costs O(1) however many fragments the node holds.

        Returns:
            tuple[bytes, bytes, int]: ``b""``, the 32-byte root, and the
            fragment count.
        """
        digest = self.node.digest
        return b"", digest.root(), len(digest)

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
                except NetworkError as exc:
                    logger.warning("gossip to %s failed (network): %s", target.node_id, exc)
                except (SchemaError, AuthError) as exc:
                    logger.warning("gossip to %s rejected (typed): %s", target.node_id, exc)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("gossip to %s failed (unexpected): %s", target.node_id, exc)
            self.stop_event.wait(timeout=self.config.gossip_interval_sec)

    def handle(self, data: JsonDict) -> JsonDict:
        """Apply an incoming gossip payload to local state.

        Merges the sender's peers, sampled fragment locations, and
        tombstones. Missing replicas are healed by the replicator's
        bucket-by-bucket repair, not here.

        Args:
            data: Incoming gossip payload (parsed JSON).

        Returns:
            JsonDict: Local gossip state for the caller.
        """
        try:
            incoming = GossipState.from_json(data)
        except SchemaError as exc:
            logger.warning("Failed to parse gossip state (schema): %s", exc)
            return {}
        except Exception as exc:
            logger.warning("Failed to parse gossip state (unexpected): %s", exc)
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
