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
        cluster_epoch: Last cluster epoch this peer advertised;
            lets a recovering node refuse to merge a stale peer
            view. ``0`` when the peer has never published an epoch.
        peer_cn: Common Name from the peer's mTLS client cert.
            ``""`` when the cluster does not enforce mTLS.
            Captured at join time so the membership table records
            which peer is which (and so a forged peer cannot
            impersonate an existing CN).
        lease_until: Wall-clock deadline after which the lease
            expires. Refreshed to ``now() + ClusterConfig.lease_timeout_sec``
            on every successful heartbeat. ``0.0`` means no
            explicit lease is held (single-node deployments).
            Phase 4 uses this field as the canonical source of
            truth for the membership table freshness, in tandem
            with the heartbeat-miss counter.
    """

    node_id: str
    host: str
    port: int
    last_heartbeat: float = 0.0
    healthy: bool = True
    suspect: bool = False
    missed_heartbeats: int = 0
    cluster_epoch: int = 0
    peer_cn: str = ""
    lease_until: float = 0.0

    def to_json(self) -> dict[str, Any]:
        """Serialize this peer to a JSON-compatible dict."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "healthy": self.healthy,
            "suspect": self.suspect,
            "missed_heartbeats": self.missed_heartbeats,
            "cluster_epoch": self.cluster_epoch,
            "peer_cn": self.peer_cn,
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

    def add(
        self,
        node_id: str,
        host: str,
        port: int,
        peer_cn: str = "",
    ) -> None:
        """Add or update a peer.

        No-op when ``node_id`` matches the local node.

        Args:
            node_id: Peer node identifier.
            host: Peer host.
            port: Peer port.
            peer_cn: Common Name from the peer's mTLS client
                cert. ``""`` when the cluster does not enforce
                mTLS. Stored on the
                :class:`~membrane.network.membership.PeerInfo`
                record so a forged peer cannot impersonate an
                existing CN at join time.
        """
        with self.lock:
            if node_id == self.node_id:
                return
            if node_id in self.peers:
                self.peers[node_id].host = host
                self.peers[node_id].port = port
                self.peers[node_id].peer_cn = peer_cn
                return
            self.peers[node_id] = PeerInfo(
                node_id=node_id,
                host=host,
                port=port,
                last_heartbeat=time.time(),
                peer_cn=peer_cn,
            )
            self.clients[node_id] = Peer(f"http://{host}:{port}")
            self.ring.add_node(node_id)
            self.shard.add_node(node_id)
            logger.info("Added peer %s at %s:%s (cn=%s)", node_id, host, port, peer_cn)

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

    def save_snapshot(self) -> list[dict[str, Any]]:
        """Return a durable snapshot of the membership table.

        The shape matches :meth:`to_json` so callers persist either
        form interchangeably.

        Returns:
            list[dict[str, Any]]: List of ``PeerInfo.to_json()``
            entries, in insertion order.
        """
        return self.to_json()

    def load_snapshot(self, payload: list[dict[str, Any]]) -> None:
        """Restore membership from a previously persisted snapshot.

        Existing entries are dropped first so callers can safely
        run this on a fresh ``Membership`` or on one that already
        holds gossip-driven entries. The local ``node_id`` is
        never included.

        Args:
            payload: Snapshot body as produced by
                :meth:`save_snapshot`.
        """
        with self.lock:
            for peer_id in list(self.peers.keys()):
                self.remove(peer_id)
            for entry in payload:
                # 1.0.x snapshots predated the mTLS field. Older
                # payloads will lack "peer_cn"; treat as empty so
                # the cluster can rebuild from the legacy file
                # without operator intervention.
                peer_cn = str(entry.get("peer_cn", ""))
                self.add(
                    node_id=entry["node_id"],
                    host=entry["host"],
                    port=int(entry["port"]),
                    peer_cn=peer_cn,
                )
                p = self.peers[entry["node_id"]]
                p.cluster_epoch = int(entry.get("cluster_epoch", 0))
                p.healthy = bool(entry.get("healthy", True))
                p.suspect = bool(entry.get("suspect", False))
                p.missed_heartbeats = int(entry.get("missed_heartbeats", 0))

    def record_heartbeat(
        self,
        node_id: str,
        lease_until: float = 0.0,
    ) -> None:
        """Reset heartbeat counters for ``node_id``.

        Optionally stamp the lease deadline returned by the
        peer (Phase 4). ``lease_until <= 0`` is ignored so
        peers that do not advertise leases keep ``lease_until = 0``
        and rely on the missed-heartbeats counter only.

        Args:
            node_id: Peer identifier whose counters to reset.
            lease_until: Optional Unix deadline.
        """
        with self.lock:
            peer = self.peers.get(node_id)
            if peer is None:
                return
            peer.last_heartbeat = time.time()
            peer.missed_heartbeats = 0
            peer.suspect = False
            peer.healthy = True
            if lease_until > 0:
                peer.lease_until = lease_until

    def evict_expired_leases(self, now: float | None = None) -> list[str]:
        """Mark peers whose lease deadline has elapsed as suspect.

        Returns the list of peer ids that crossed into the
        ``suspect`` state during this scan. The caller (typically
        :class:`~membrane.network.cluster.Cluster`) can chain
        the standard failure-removal pass to evict them.

        Args:
            now: Wall-clock now for deterministic testing;
                ``None`` reads ``time.time()`` at call time.

        Returns:
            list[str]: Newly-suspect peer ids.
        """
        if now is None:
            now = time.time()
        flagged: list[str] = []
        with self.lock:
            for peer_id, peer in self.peers.items():
                if (
                    peer.lease_until > 0
                    and peer.healthy
                    and now > peer.lease_until
                    and self.mark_suspect(peer_id)
                ):
                    flagged.append(peer_id)
        return flagged

    def leave_cluster(self, local_node_id: str | None = None) -> int:
        """Best-effort graceful leave. Fires ``POST /leave`` to every healthy peer.

        Returns the number of peers contacted; transport failures
        are logged and skipped so a partial cluster can still
        process the leave without blocking shutdown.

        Args:
            local_node_id: Identifier of the leaving node. When
                ``None`` we read ``self.node_id``.

        Returns:
            int: Number of peers the leave was dispatched to.
        """
        leaving = local_node_id or self.node_id
        dispatched = 0
        with self.lock:
            peers = list(self.peers.values())
        for peer in peers:
            client = self.get_client(peer.node_id)
            if client is None:
                continue
            try:
                if client.leave_cluster(leaving):
                    dispatched += 1
            except Exception as exc:
                logger.debug(
                    "leave_cluster delivery to %s failed: %s",
                    peer.node_id,
                    exc,
                )
        return dispatched

    def record_peer_cn(self, node_id: str, cn: str) -> None:
        """Stamp the verified peer cert CN onto a membership record.

        Called from :func:`op_heartbeat` and :func:`op_join` every
        time a peer proves its identity. ``""`` clears the
        recorded value when the cluster is not enforcing mTLS
        for that particular peer.

        Args:
            node_id: Identifier of the peer whose entry is
                updated.
            cn: The verified Common Name from the peer cert, or
                ``""`` to clear the recorded identity.
        """
        with self.lock:
            peer = self.peers.get(node_id)
            if peer is None:
                return
            peer.peer_cn = cn
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

    def join_seeds(
        self,
        seeds: list[str],
        local_node_id: str,
        host: str,
        port: int,
    ) -> bool:
        """Contact each seed peer in order; stop at the first success.

        On success, the response payload's peers are merged
        into the local membership table (excluding the local
        node id, which the seed may have echoed back).

        Args:
            seeds: Seed peer URLs as ``"host:port"`` strings.
            local_node_id: Local node identifier.
            host: Local host (echoed to the seed so it knows
                how to reach us).
            port: Local port.

        Returns:
            bool: ``True`` when at least one seed accepted the join.
        """
        for seed in seeds:
            try:
                client = Peer(f"http://{seed}")
                result = client.join_cluster(local_node_id, host, port)
            except Exception as exc:
                logger.warning("Bootstrap failed for seed %s: %s", seed, exc)
                continue
            if result and result.get("success"):
                for peer in result.get("peers", []):
                    self.add(peer["node_id"], peer["host"], peer["port"])
                logger.info("Bootstrap successful via %s", seed)
                return True
        return False


__all__ = ["Membership", "PeerInfo"]
