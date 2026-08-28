"""Gossip state for peer-to-peer anti-entropy protocol.

This module defines the two dataclasses that flow through
Membrane's gossip protocol:

* :class:`PeerEndpoint` — a peer's network address and health.
* :class:`GossipState` — the full payload exchanged between
  peers during a gossip round, including membership, fragment
  location samples, and an inventory digest.

Both classes provide :meth:`to_json` and :meth:`from_json`
helpers so they can be serialized with the standard library
without any extra dependency.
"""

from dataclasses import dataclass, field
from typing import Any


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

    def to_json(self) -> dict[str, Any]:
        """Serialize this endpoint to a JSON-compatible dict.

        Returns:
            dict[str, Any]: ``node_id``, ``host``, ``port``,
            ``healthy``.
        """
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PeerEndpoint":
        """Deserialize a peer endpoint from a JSON-compatible dict.

        Args:
            data: Mapping previously produced by
                :meth:`to_json`.

        Returns:
            PeerEndpoint: Reconstructed instance.
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

    A gossip state bundles four pieces of information:

    * ``peers`` — the sender's current membership view.
    * ``fragment_locations`` — a sampled subset of the
      ``content_hash -> [node_ids]`` mapping the sender knows
      about. Sampling bounds the message size; receivers fill
      in the rest via additional rounds.
    * ``inventory_digest`` — ``content_hash -> version_id`` for
      every fragment the sender holds locally. Used by
      :class:`~membrane.delta_sync.DeltaSync` to compute missing
      or outdated entries on the receiving side.
    * ``timestamp`` — sender's wall-clock time at emission;
      receivers can use it to discard stale states.

    Attributes:
        node_id: Sender's node identifier.
        timestamp: Unix timestamp at emission.
        peers: List of known peer endpoints.
        fragment_locations: Sampled fragment-location map.
        inventory_digest: Full inventory digest.
    """

    node_id: str
    timestamp: float
    peers: list[PeerEndpoint] = field(default_factory=list)
    fragment_locations: dict[str, list[str]] = field(default_factory=dict)
    inventory_digest: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize this state to a JSON-compatible dict.

        Returns:
            dict[str, Any]: ``node_id``, ``timestamp``,
            ``peers`` (each serialized via
            :meth:`PeerEndpoint.to_json`), ``fragment_locations``,
            and ``inventory_digest``.
        """
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "peers": [p.to_json() for p in self.peers],
            "fragment_locations": self.fragment_locations,
            "inventory_digest": self.inventory_digest,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GossipState":
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
        )

    def merge(self, other: "GossipState") -> "GossipState":
        """Merge another gossip state into a new combined state.

        Peer entries are de-duplicated by ``node_id`` with a
        small heuristic that prefers the healthier endpoint when
        both are present. Fragment-location lists are unioned.
        Inventory digest entries from ``other`` overwrite
        entries from ``self`` (each fragment is owned by exactly
        one node, so the latest observation wins).

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

        # Inventory digest is overwritten by the other side; it
        # represents the *other* node's own view of what it
        # holds locally.
        merged_digest = dict(self.inventory_digest)
        merged_digest.update(other.inventory_digest)

        return GossipState(
            node_id=self.node_id,
            timestamp=max(self.timestamp, other.timestamp),
            peers=list(merged_peers.values()),
            fragment_locations=merged_locations,
            inventory_digest=merged_digest,
        )
