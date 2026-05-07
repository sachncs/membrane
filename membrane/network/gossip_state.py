"""Gossip state for peer-to-peer anti-entropy protocol."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PeerEndpoint:
    """Network endpoint of a Membrane peer."""

    node_id: str
    host: str
    port: int
    healthy: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PeerEndpoint":
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            healthy=data.get("healthy", True),
        )


@dataclass
class GossipState:
    """Serializable state exchanged during gossip rounds."""

    node_id: str
    timestamp: float
    peers: list[PeerEndpoint] = field(default_factory=list)
    fragment_locations: dict[str, list[str]] = field(default_factory=dict)
    inventory_digest: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "peers": [p.to_json() for p in self.peers],
            "fragment_locations": self.fragment_locations,
            "inventory_digest": self.inventory_digest,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GossipState":
        return cls(
            node_id=data["node_id"],
            timestamp=data["timestamp"],
            peers=[PeerEndpoint.from_json(p) for p in data.get("peers", [])],
            fragment_locations=dict(data.get("fragment_locations", {})),
            inventory_digest=dict(data.get("inventory_digest", {})),
        )

    def merge(self, other: "GossipState") -> "GossipState":
        """Merge another gossip state into this one.

        Returns a new GossipState with combined peer list and fragment locations.
        """
        merged_peers = {p.node_id: p for p in self.peers}
        for p in other.peers:
            if p.node_id not in merged_peers:
                merged_peers[p.node_id] = p
            else:
                # Prefer the newer state for the same peer
                existing = merged_peers[p.node_id]
                if not existing.healthy and p.healthy:
                    merged_peers[p.node_id] = p

        merged_locations = dict(self.fragment_locations)
        for h, nodes in other.fragment_locations.items():
            current_nodes = set(merged_locations.get(h, []))
            current_nodes.update(nodes)
            merged_locations[h] = list(current_nodes)

        merged_digest = dict(self.inventory_digest)
        merged_digest.update(other.inventory_digest)

        return GossipState(
            node_id=self.node_id,
            timestamp=max(self.timestamp, other.timestamp),
            peers=list(merged_peers.values()),
            fragment_locations=merged_locations,
            inventory_digest=merged_digest,
        )
