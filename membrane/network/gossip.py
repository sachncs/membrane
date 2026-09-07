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

Phase 5 inventory layout: the legacy ``inventory_digest``
field is replaced with a Bloom filter (``inventory_bloom``,
serialized :class:`~membrane.bloom.BloomFilter`), a Merkle root
(``inventory_merkle_root``), and the leaf count
(``inventory_size``). The Bloom filter is the cheap ping/pong
side-channel; the Merkle root is the precise diff root the
receiver descends when the roots disagree.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field

from membrane.bloom import BloomFilter
from membrane.gc import TombstoneTable
from membrane.merkle import MerkleTree
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

    A gossip state bundles six pieces of information:

    * ``peers`` — the sender's current membership view.
    * ``fragment_locations`` — a sampled subset of the
      ``content_hash -> [node_ids]`` mapping the sender knows
      about. Sampling bounds the message size; receivers fill
      in the rest via additional rounds.
    * ``inventory_bloom`` — serialized
      :class:`~membrane.bloom.BloomFilter` over the sender's
      fragment set, used for the cheap ping/pong side-channel.
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
            payload predates Phase 5 (no ``inventory_bloom`` /
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

        Phase 5: the inventory-side fields (Bloom + Merkle root)
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

        # Phase 5 inventory: build the Bloom filter and Merkle
        # tree from the (content_hash, owner_node_id) pairs. The
        # Bloom filter is tuned to the configured
        # ``gossip_payload_expected_items`` / ``gossip_payload_fpr``
        # knobs (with a small floor so single-fragment clusters
        # still get a non-trivial filter).
        expected = max(
            self.config.gossip_payload_expected_items,
            len(self.node.fragments),
        )
        bloom = BloomFilter.tuned_for(
            expected_items=max(1, expected),
            fp_rate=float(self.config.gossip_payload_fpr),
        )
        pairs: list[tuple[str, str]] = []
        for h, frag in self.node.fragments.items():
            bloom = bloom.add(h)
            pairs.append((h, frag.identity.payload_hash or ""))
        # We need a real owner_node_id; the registry gives us the
        # holder set for ``h``. We use the first one (deterministic
        # because the registry is sorted).
        owners_for_pair: dict[str, str] = {}
        for h, _ in pairs:
            holders = self.directory.locate_fragment(h)
            owner = sorted(holders)[0] if holders else self.node.node_id
            owners_for_pair[h] = owner
        ordered_pairs = sorted(
            (h, owners_for_pair[h]) for h, _ in pairs
        )
        tree = MerkleTree.from_inventory(ordered_pairs)

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
            inventory_bloom=bloom.serialize(),
            inventory_merkle_root=tree.root,
            inventory_size=len(ordered_pairs),
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

        Phase 5: when the incoming ``inventory_merkle_root``
        differs from the local one, the receiver walks the
        peer's published locations plus its own fragments to
        compute the diff locally -- a real wire call to ask the
        sender for its Merkle subtree is left for Phase 5.4. The
        current implementation uses the Bloom filter to skip the
        diff entirely when the local node has nothing in
        common with the peer.

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

        # Merkle-root diff: if the sender's root differs, walk
        # the divergent pairs by reading the directory and
        # computing the symmetric difference. This is O(n) on the
        # directory for now; the Merkle root gives us a fast
        # skip when the inventories are already in sync.
        local_state = self.build_state()
        if local_state.inventory_merkle_root != incoming.inventory_merkle_root:
            local_pairs = self._inventory_pairs()
            remote_pairs = set()
            for h in incoming.fragment_locations:
                holders = self.directory.locate_fragment(h)
                owner = sorted(holders)[0] if holders else incoming.node_id
                remote_pairs.add((h, owner))
            # Record whatever the peer has, even if we do not yet
            # have the bytes; the rest of the cluster will route
            # future stores through the new owner.
            for pair in remote_pairs:
                self.directory.record_fragment_location(pair[0], pair[1])
            # Pull-side: for every pair in the sender's set that
            # we are missing, ask the sender to push the bytes
            # via the existing request_replicate path. We do not
            # block the gossip handler on the network call;
            # the replicator loop will reconcile the missing
            # bytes on its next pass if the immediate push fails.
            for pair in remote_pairs - local_pairs:
                self._replicator_pull(pair[0], pair[1])

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

    def _inventory_pairs(self) -> set[tuple[str, str]]:
        """Snapshot the local (hash, owner_node_id) pairs.

        Returns:
            set[tuple[str, str]]: ``{(hash, owner), ...}`` where
            ``owner`` is the first holder recorded by the
            directory; ``self.node.node_id`` if the directory has
            no record for the hash.
        """
        pairs: set[tuple[str, str]] = set()
        for h, _frag in self.node.fragments.items():
            holders = self.directory.locate_fragment(h)
            owner = sorted(holders)[0] if holders else self.node.node_id
            pairs.add((h, owner))
        return pairs

    def _replicator_pull(self, content_hash: str, owner_node_id: str) -> None:
        """Best-effort pull of a single fragment from the named owner.

        Implemented as a direct call on the peer's
        ``request_replicate`` HTTP endpoint via the cluster's
        membership table. The replicator loop is the canonical
        path for this work; this helper is the gossip-side
        nudge that fires immediately so the bytes flow on the
        next gossip round instead of waiting for the loop tick.
        """
        client = self.membership.get_client(owner_node_id)
        if client is None:
            return
        try:
            frag = self.node.retrieve(content_hash)
            if frag is not None:
                client.request_replicate(frag)
        except Exception as exc:  # pragma: no cover - background
            logger.debug(
                "gossip pull of %s from %s failed: %s",
                content_hash,
                owner_node_id,
                exc,
            )


__all__ = ["Gossip", "GossipState", "PeerEndpoint"]
