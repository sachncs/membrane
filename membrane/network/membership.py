"""Membership: thread-safe cluster-membership table.

Owns the ``node_id -> PeerInfo`` map, the cached HTTP clients, and
the membership mutations (add/remove). Every other cluster subsystem
queries membership through this class; no one else holds a
reference to the membership dict.

Concurrency:
    All public methods are protected by an internal
    :class:`threading.RLock` and are safe to call from multiple
    threads.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from membrane.network.peer import Peer
from membrane.registry import Registry
from membrane.ring import Ring
from membrane.shard import Shard

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Runtime state for a known peer.

    Attributes:
        node_id: Peer node identifier.
        host: Peer host.
        port: Peer port.
        last_heartbeat: Unix timestamp of the most recent
            successful heartbeat.
        healthy: Whether the peer is currently considered healthy.
        suspect: Whether the peer has crossed the suspect
            threshold but not yet the remove threshold.
        missed_heartbeats: Counter of consecutive failed heartbeats.
    """

    node_id: str
    host: str
    port: int
    last_heartbeat: float = 0.0
    healthy: bool = True
    suspect: bool = False
    missed_heartbeats: int = 0

    def to_json(self) -> dict[str, Any]:
        """Serialize this peer to a JSON-compatible dict."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
            "suspect": self.suspect,
            "missed_heartbeats": self.missed_heartbeats,
        }


class Membership:
    """Thread-safe cluster-membership table.

    Args:
        node_id: Identifier of the local node; this node is never
            included in its own membership view.
        ring: Consistent-hash ring; updated on add/remove.
        shard: Shard manager; updated on add/remove.
        directory: Optional fragment-location directory; cleaned up
            on remove.
    """

    def __init__(
        self,
        node_id: str,
        ring: Ring,
        shard: Shard,
        directory: Registry | None = None,
    ) -> None:
        self.node_id = node_id
        self.ring = ring
        self.shard = shard
        self.directory = directory or Registry()

        self.peers: dict[str, PeerInfo] = {}
        self.clients: dict[str, Peer] = {}
        self.lock = threading.RLock()

    def add(self, node_id: str, host: str, port: int) -> None:
        """Add or update a peer.

        No-op when ``node_id`` matches the local node.

        Args:
            node_id: Peer node identifier.
            host: Peer host.
            port: Peer port.
        """
        with self.lock:
            if node_id == self.node_id:
                return
            if node_id in self.peers:
                self.peers[node_id].host = host
                self.peers[node_id].port = port
                return
            self.peers[node_id] = PeerInfo(
                node_id=node_id, host=host, port=port, last_heartbeat=time.time()
            )
            self.clients[node_id] = Peer(f"http://{host}:{port}")
            self.ring.add_node(node_id)
            self.shard.add_node(node_id)
            logger.info("Added peer %s at %s:%s", node_id, host, port)

    def remove(self, node_id: str) -> bool:
        """Remove a peer; return True when it was registered."""
        with self.lock:
            if node_id not in self.peers:
                return False
            del self.peers[node_id]
            self.clients.pop(node_id, None)
            self.ring.remove_node(node_id)
            self.shard.remove_node(node_id)
            self.directory.unregister_node(node_id)
            logger.info("Removed peer %s", node_id)
            return True

    def snapshot(self) -> list[PeerInfo]:
        """Return a copy of the membership list."""
        with self.lock:
            return list(self.peers.values())

    def find(self, node_id: str) -> PeerInfo | None:
        """Return the PeerInfo for ``node_id`` or None."""
        with self.lock:
            return self.peers.get(node_id)

    def get_client(self, node_id: str) -> Peer | None:
        """Return the cached HTTP client for a peer."""
        with self.lock:
            return self.clients.get(node_id)

    def get_url(self, node_id: str) -> str | None:
        """Return ``http://<host>:<port>`` for a peer, or None."""
        with self.lock:
            p = self.peers.get(node_id)
            return f"http://{p.host}:{p.port}" if p else None

    def healthy(self) -> list[PeerInfo]:
        """Return the list of currently healthy peers."""
        with self.lock:
            return [p for p in self.peers.values() if p.healthy]

    def to_json(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of membership."""
        with self.lock:
            return [p.to_json() for p in self.peers.values()]

    def record_heartbeat(self, node_id: str) -> None:
        """Reset heartbeat counters for ``node_id``.

        No-op when the peer is unknown.
        """
        with self.lock:
            p = self.peers.get(node_id)
            if p is None:
                return
            p.last_heartbeat = time.time()
            p.missed_heartbeats = 0
            p.suspect = False
            p.healthy = True

    def record_miss(self, node_id: str) -> None:
        """Increment the missed-heartbeat counter for ``node_id``."""
        with self.lock:
            p = self.peers.get(node_id)
            if p is not None:
                p.missed_heartbeats += 1

    def mark_suspect(self, node_id: str) -> bool:
        """Mark ``node_id`` as suspect. Returns True on transition."""
        with self.lock:
            p = self.peers.get(node_id)
            if p is None or p.suspect:
                return False
            p.suspect = True
            return True


__all__ = ["Membership", "PeerInfo"]
