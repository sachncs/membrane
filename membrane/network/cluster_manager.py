"""ClusterManager: peer-to-peer bootstrap, heartbeat, gossip, and replication.

Runs background daemon threads for membership maintenance and state exchange.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from membrane.global_directory import GlobalDirectory
from membrane.hash_ring import HashRing
from membrane.membrane_node import MembraneNode
from membrane.network.config import ClusterConfig
from membrane.network.gossip_state import GossipState, PeerEndpoint
from membrane.network.peer_client import PeerClient
from membrane.shard_manager import ShardManager

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Runtime state for a known peer."""

    node_id: str
    host: str
    port: int
    last_heartbeat: float = 0.0
    healthy: bool = True
    suspect: bool = False
    missed_heartbeats: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
            "suspect": self.suspect,
            "missed_heartbeats": self.missed_heartbeats,
        }


class ClusterManager:
    """Coordinates cluster membership, gossip, and replication.

    Args:
        node_id: Identifier for this node.
        host: Bind host.
        port: Listen port.
        node: Local MembraneNode.
        config: Cluster configuration.
        directory: Optional GlobalDirectory.
        hash_ring: Optional HashRing.
        shard_manager: Optional ShardManager.
    """

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        node: MembraneNode,
        config: ClusterConfig,
        directory: GlobalDirectory | None = None,
        hash_ring: HashRing | None = None,
        shard_manager: ShardManager | None = None,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.node = node
        self.config = config
        self.directory = directory or GlobalDirectory()
        self.hash_ring = hash_ring or HashRing()
        self.shard_manager = shard_manager or ShardManager(self.hash_ring)

        self._peers: dict[str, PeerInfo] = {}
        self._clients: dict[str, PeerClient] = {}
        self._lock = threading.RLock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background threads."""
        self._running = True
        self._stop_event.clear()

        loops = [
            (self._bootstrap_loop, "bootstrap"),
            (self._heartbeat_loop, "heartbeat"),
            (self._failure_detection_loop, "failure-detection"),
        ]
        if self.config.enable_gossip:
            loops.append((self._gossip_loop, "gossip"))
        if self.config.enable_replication:
            loops.append((self._replication_loop, "replication"))

        for target, name in loops:
            t = threading.Thread(target=target, daemon=True, name=f"membrane-{name}")
            t.start()
            self._threads.append(t)

        logger.info("ClusterManager started with %s background threads", len(loops))

    def stop(self) -> None:
        """Signal all background threads to exit."""
        self._running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        logger.info("ClusterManager stopped")

    def join(self) -> None:
        """Block until stop() is called."""
        self._stop_event.wait()

    # ------------------------------------------------------------------
    # Membership API (thread-safe)
    # ------------------------------------------------------------------

    def add_peer(self, node_id: str, host: str, port: int) -> None:
        with self._lock:
            if node_id == self.node_id:
                return
            if node_id in self._peers:
                # Update endpoint if changed
                self._peers[node_id].host = host
                self._peers[node_id].port = port
                return
            self._peers[node_id] = PeerInfo(
                node_id=node_id, host=host, port=port, last_heartbeat=time.time()
            )
            self._clients[node_id] = PeerClient(f"http://{host}:{port}")
            self.hash_ring.add_node(node_id)
            self.shard_manager.add_node(node_id)
            logger.info("Added peer %s at %s:%s", node_id, host, port)

    def remove_peer(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._peers:
                return False
            del self._peers[node_id]
            self._clients.pop(node_id, None)
            self.hash_ring.remove_node(node_id)
            self.shard_manager.remove_node(node_id)
            self.directory.unregister_node(node_id)
            logger.info("Removed peer %s", node_id)
            return True

    def get_peers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [p.to_json() for p in self._peers.values()]

    def is_peer_healthy(self, node_id: str) -> bool:
        with self._lock:
            p = self._peers.get(node_id)
            return p.healthy if p else False

    def get_peer_client(self, node_id: str) -> PeerClient | None:
        with self._lock:
            return self._clients.get(node_id)

    def get_peer_url(self, node_id: str) -> str | None:
        with self._lock:
            p = self._peers.get(node_id)
            if p:
                return f"http://{p.host}:{p.port}"
            return None

    # ------------------------------------------------------------------
    # Event handlers (called by HTTP server)
    # ------------------------------------------------------------------

    def on_peer_join(self, node_id: str, host: str, port: int) -> dict[str, Any]:
        self.add_peer(node_id, host, port)
        with self._lock:
            peers = [
                {"node_id": p.node_id, "host": p.host, "port": p.port}
                for p in self._peers.values()
            ]
        return {"success": True, "peers": peers}

    def on_peer_leave(self, node_id: str) -> None:
        self.remove_peer(node_id)

    def on_heartbeat(self, node_id: str) -> dict[str, Any]:
        with self._lock:
            p = self._peers.get(node_id)
            if p:
                p.last_heartbeat = time.time()
                p.missed_heartbeats = 0
                p.suspect = False
                p.healthy = True
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
        try:
            incoming = GossipState.from_json(data)
        except Exception as exc:
            logger.warning("Failed to parse gossip state: %s", exc)
            return {}

        # Add/update peers from incoming state
        for ep in incoming.peers:
            if ep.node_id != self.node_id:
                self.add_peer(ep.node_id, ep.host, ep.port)

        # Merge fragment locations into directory
        for h, nodes in incoming.fragment_locations.items():
            for nid in nodes:
                self.directory.record_fragment_location(h, nid)

        # Build response with our state
        with self._lock:
            our_peers = [
                PeerEndpoint(
                    node_id=p.node_id,
                    host=p.host,
                    port=p.port,
                    healthy=p.healthy,
                )
                for p in self._peers.values()
            ]
            # Sample fragment locations
            all_hashes = list(self.node.fragments.keys())
            sample_hashes = random.sample(
                all_hashes,
                min(self.config.gossip_max_fragment_entries, len(all_hashes)),
            )
            fragment_locations: dict[str, list[str]] = {}
            for h in sample_hashes:
                fragment_locations[h] = list(self.directory.locate_fragment(h))
            digest = {h: frag.version_id for h, frag in self.node.fragments.items()}

        response = GossipState(
            node_id=self.node_id,
            timestamp=time.time(),
            peers=our_peers,
            fragment_locations=fragment_locations,
            inventory_digest=digest,
        )
        return response.to_json()

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _bootstrap_loop(self) -> None:
        """Contact seed peers and join the cluster."""
        for seed in self.config.peers:
            if self._stop_event.is_set():
                return
            try:
                client = PeerClient(f"http://{seed}")
                result = client.join_cluster(self.node_id, self.host, self.port)
                if result and result.get("success"):
                    for peer in result.get("peers", []):
                        self.add_peer(peer["node_id"], peer["host"], peer["port"])
                    logger.info("Bootstrap successful via %s", seed)
                    break
            except Exception as exc:
                logger.warning("Bootstrap failed for seed %s: %s", seed, exc)

    def _heartbeat_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            with self._lock:
                peers = list(self._peers.values())
            for p in peers:
                if self._stop_event.is_set():
                    return
                client = self.get_peer_client(p.node_id)
                if client is None:
                    continue
                try:
                    resp = client.heartbeat()
                    if resp:
                        with self._lock:
                            peer = self._peers.get(p.node_id)
                            if peer:
                                peer.last_heartbeat = time.time()
                                peer.missed_heartbeats = 0
                                peer.suspect = False
                                peer.healthy = True
                except Exception as exc:
                    with self._lock:
                        peer = self._peers.get(p.node_id)
                        if peer:
                            peer.missed_heartbeats += 1
                    logger.debug("Heartbeat to %s failed: %s", p.node_id, exc)
            self._stop_event.wait(timeout=self.config.heartbeat_interval_sec)

    def _failure_detection_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            now = time.time()
            to_remove: list[str] = []
            with self._lock:
                for node_id, p in list(self._peers.items()):
                    if p.missed_heartbeats >= self.config.failure_remove_threshold:
                        to_remove.append(node_id)
                    elif p.missed_heartbeats >= self.config.failure_suspect_threshold:
                        if not p.suspect:
                            p.suspect = True
                            logger.warning("Peer %s is now suspect", node_id)
            for node_id in to_remove:
                logger.warning("Removing failed peer %s", node_id)
                self.remove_peer(node_id)
            self._stop_event.wait(timeout=self.config.heartbeat_interval_sec)

    def _gossip_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            with self._lock:
                healthy_peers = [p for p in self._peers.values() if p.healthy]
            if not healthy_peers:
                self._stop_event.wait(timeout=self.config.gossip_interval_sec)
                continue

            targets = random.sample(
                healthy_peers,
                min(self.config.gossip_fanout, len(healthy_peers)),
            )

            with self._lock:
                our_peers = [
                    PeerEndpoint(p.node_id, p.host, p.port, p.healthy)
                    for p in self._peers.values()
                ]
                all_hashes = list(self.node.fragments.keys())
                sample_hashes = random.sample(
                    all_hashes,
                    min(self.config.gossip_max_fragment_entries, len(all_hashes)),
                )
                fragment_locations: dict[str, list[str]] = {}
                for h in sample_hashes:
                    fragment_locations[h] = list(self.directory.locate_fragment(h))
                digest = {h: frag.version_id for h, frag in self.node.fragments.items()}

            state = GossipState(
                node_id=self.node_id,
                timestamp=time.time(),
                peers=our_peers,
                fragment_locations=fragment_locations,
                inventory_digest=digest,
            )

            for target in targets:
                if self._stop_event.is_set():
                    return
                client = self.get_peer_client(target.node_id)
                if client is None:
                    continue
                try:
                    resp = client.gossip(state.to_json())
                    if resp:
                        self.on_gossip(resp)
                except Exception as exc:
                    logger.debug("Gossip to %s failed: %s", target.node_id, exc)

            self._stop_event.wait(timeout=self.config.gossip_interval_sec)

    def _replication_loop(self) -> None:
        """Background replication of missing fragments."""
        while self._running and not self._stop_event.is_set():
            with self._lock:
                healthy_peers = [p for p in self._peers.values() if p.healthy]
                primary_hashes = list(self.node.get_shard_hashes())

            for h in primary_hashes:
                if self._stop_event.is_set():
                    return
                replicas = self.shard_manager.get_replicas(h)
                for peer_id in replicas:
                    if peer_id == self.node_id:
                        continue
                    # Check if peer already has it
                    client = self.get_peer_client(peer_id)
                    if client is None:
                        continue
                    try:
                        existing = client.retrieve_fragment(h)
                        if existing is None:
                            frag = self.node.retrieve(h)
                            if frag:
                                client.request_replicate(frag)
                                logger.debug("Replicated %s to %s", h, peer_id)
                    except Exception as exc:
                        logger.debug("Replication of %s to %s failed: %s", h, peer_id, exc)

            self._stop_event.wait(timeout=self.config.gossip_interval_sec)
