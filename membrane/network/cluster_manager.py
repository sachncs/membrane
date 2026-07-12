"""ClusterManager: peer-to-peer bootstrap, heartbeat, gossip, and replication.

Runs background daemon threads for membership maintenance and
state exchange.

The :class:`ClusterManager` is the heart of Membrane's runtime:
it owns the cluster-membership tables, the
:class:`~membrane.hash_ring.HashRing`,
:class:`~membrane.shard_manager.ShardManager`, and
:class:`~membrane.global_directory.GlobalDirectory`; runs the
periodic bootstrap, heartbeat, failure-detection, gossip, and
replication loops; and exposes a small synchronous API for
membership queries and event handling.

Threading:
    * Membership mutations are protected by an internal
      :class:`threading.RLock`.
    * The background loops run as daemon threads; they are
      stopped by :meth:`stop` (which sets ``_stop_event`` and
      joins each thread with a short timeout).
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
    """Runtime state for a known peer.

    Attributes:
        node_id: Peer node identifier.
        host: Peer host.
        port: Peer port.
        last_heartbeat: Unix timestamp of the most recent
            successful heartbeat.
        healthy: Whether the peer is currently considered
            healthy.
        suspect: Whether the peer has crossed the suspect
            threshold but not yet the remove threshold.
        missed_heartbeats: Counter of consecutive failed
            heartbeats.
    """

    node_id: str
    host: str
    port: int
    last_heartbeat: float = 0.0
    healthy: bool = True
    suspect: bool = False
    missed_heartbeats: int = 0

    def to_json(self) -> dict[str, Any]:
        """Serialize this peer to a JSON-compatible dict.

        Returns:
            dict[str, Any]: ``node_id``, ``host``, ``port``,
            ``healthy``, ``suspect``, and ``missed_heartbeats``.
        """
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
        node: Local :class:`MembraneNode`.
        config: Cluster configuration.
        directory: Optional :class:`GlobalDirectory`.
        hash_ring: Optional :class:`HashRing`.
        shard_manager: Optional :class:`ShardManager`.
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
        """Initialize the cluster manager and internal state."""
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
        """Start background threads.

        Launches bootstrap, heartbeat, and failure-detection
        threads unconditionally. Gossip and replication threads
        are started only when the corresponding
        ``config.enable_*`` flag is set.
        """
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
        """Signal all background threads to exit.

        Sets ``_running`` to ``False`` and ``_stop_event``, then
        joins each background thread with a short timeout.
        Threads that do not terminate within the timeout remain
        alive (they are daemon threads, so they will not block
        process exit).
        """
        self._running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        logger.info("ClusterManager stopped")

    def join(self) -> None:
        """Block until :meth:`stop` is called.

        Useful as a foreground companion to ``start()`` when the
        caller wants the manager to live for the duration of the
        process.
        """
        self._stop_event.wait()

    # ------------------------------------------------------------------
    # Membership API (thread-safe)
    # ------------------------------------------------------------------

    def add_peer(self, node_id: str, host: str, port: int) -> None:
        """Add or update a peer in the membership table.

        Args:
            node_id: Peer node identifier.
            host: Peer host.
            port: Peer port.
        """
        with self._lock:
            if node_id == self.node_id:
                # Never add ourselves.
                return
            if node_id in self._peers:
                # Refresh endpoint if changed.
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
        """Remove a peer from the cluster.

        Args:
            node_id: Peer node identifier.

        Returns:
            bool: True when the peer was registered and is
            now removed, False if it was unknown.
        """
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
        """Return a snapshot of the membership table.

        Returns:
            list[dict[str, Any]]: Per-peer
            :meth:`PeerInfo.to_json` snapshots.
        """
        with self._lock:
            return [p.to_json() for p in self._peers.values()]

    def is_peer_healthy(self, node_id: str) -> bool:
        """Return whether a peer is currently healthy.

        Args:
            node_id: Peer node identifier.

        Returns:
            bool: True when the peer exists and is healthy,
            False otherwise.
        """
        with self._lock:
            p = self._peers.get(node_id)
            return p.healthy if p else False

    def get_peer_client(self, node_id: str) -> PeerClient | None:
        """Return the cached HTTP client for a peer.

        Args:
            node_id: Peer node identifier.

        Returns:
            PeerClient | None: The cached client, or ``None``
            if the peer is unknown.
        """
        with self._lock:
            return self._clients.get(node_id)

    def get_peer_url(self, node_id: str) -> str | None:
        """Return the HTTP base URL for a peer.

        Args:
            node_id: Peer node identifier.

        Returns:
            str | None: ``http://<host>:<port>`` or ``None``.
        """
        with self._lock:
            p = self._peers.get(node_id)
            if p:
                return f"http://{p.host}:{p.port}"
            return None

    # ------------------------------------------------------------------
    # Event handlers (called by HTTP server)
    # ------------------------------------------------------------------

    def on_peer_join(self, node_id: str, host: str, port: int) -> dict[str, Any]:
        """Handle a ``POST /join`` request from a peer.

        Args:
            node_id: Joining node's identifier.
            host: Joining node's host.
            port: Joining node's port.

        Returns:
            dict[str, Any]: ``{"success": True, "peers": [...]}``
            where ``peers`` is the current membership view
            (excluding the joiner).
        """
        self.add_peer(node_id, host, port)
        with self._lock:
            peers = [
                {"node_id": p.node_id, "host": p.host, "port": p.port}
                for p in self._peers.values()
            ]
        return {"success": True, "peers": peers}

    def on_peer_leave(self, node_id: str) -> None:
        """Handle a ``POST /leave`` request.

        Args:
            node_id: Leaving node's identifier.
        """
        self.remove_peer(node_id)

    def on_heartbeat(self, node_id: str) -> dict[str, Any]:
        """Handle a ``POST /heartbeat`` request.

        Refreshes the peer's heartbeat counters and returns the
        local node's view of its own load for the caller's
        bookkeeping.

        Args:
            node_id: Heartbeating peer's identifier.

        Returns:
            dict[str, Any]: ``node_id``, ``load``,
            ``memory_used_bytes``, ``memory_limit_bytes``,
            ``fragment_count``, ``healthy``.
        """
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
        """Handle a ``POST /gossip`` request.

        Merges the incoming peer's view into the local state and
        returns the local view as the response. Unknown or
        malformed gossip payloads yield an empty response.

        Args:
            data: Incoming gossip payload.

        Returns:
            dict[str, Any]: Local gossip state serialized via
            :meth:`GossipState.to_json`. Empty dict on parse
            failure.
        """
        try:
            incoming = GossipState.from_json(data)
        except Exception as exc:
            logger.warning("Failed to parse gossip state: %s", exc)
            return {}

        # Add/update peers from the incoming state.
        for ep in incoming.peers:
            if ep.node_id != self.node_id:
                self.add_peer(ep.node_id, ep.host, ep.port)

        # Merge fragment locations into the local directory.
        for h, nodes in incoming.fragment_locations.items():
            for nid in nodes:
                self.directory.record_fragment_location(h, nid)

        # Build response with our state.
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
            # Sample fragment locations to bound the gossip size.
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
        """Contact seed peers and join the cluster.

        Tries each configured seed in order; the first
        successful join terminates the loop. The other peers
        are then added via the response payload.
        """
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
        """Periodically ping every known peer.

        On success, the peer's ``last_heartbeat`` and
        ``missed_heartbeats`` are reset and ``healthy`` is set
        to ``True``. On failure, ``missed_heartbeats`` is
        incremented.
        """
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
        """Mark suspect peers and remove failed peers.

        Peers whose ``missed_heartbeats`` exceeds
        ``failure_suspect_threshold`` are flagged suspect.
        Peers whose count exceeds ``failure_remove_threshold``
        are removed from the membership table.
        """
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
        """Periodically push our gossip state to a random fanout.

        Each round picks up to ``gossip_fanout`` healthy peers
        uniformly at random and exchanges a sampled view of our
        fragment locations and inventory digest.
        """
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
        """Background replication of missing primary shards.

        For every primary hash held locally, ask each replica
        target whether it already has the fragment. If not,
        push the fragment via :class:`PeerClient`.
        """
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
                        # Skip ourselves.
                        continue
                    client = self.get_peer_client(peer_id)
                    if client is None:
                        continue
                    try:
                        # Ask the peer whether it already holds
                        # the fragment; if not, replicate it.
                        existing = client.retrieve_fragment(h)
                        if existing is None:
                            frag = self.node.retrieve(h)
                            if frag:
                                client.request_replicate(frag)
                                logger.debug("Replicated %s to %s", h, peer_id)
                    except Exception as exc:
                        logger.debug(
                            "Replication of %s to %s failed: %s", h, peer_id, exc
                        )

            self._stop_event.wait(timeout=self.config.gossip_interval_sec)
