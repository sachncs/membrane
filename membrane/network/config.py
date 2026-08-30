"""Cluster configuration for Membrane peer-to-peer networking.

This module defines :class:`ClusterConfig`, the single source of
truth for the runtime parameters that govern a Membrane node's
participation in a cluster: bind addresses, peer seeds, heartbeat
and gossip intervals, failure thresholds, retry policy, and
replication knobs.

Callers typically construct a :class:`ClusterConfig` once at
process start (often loading values from environment variables)
and pass it to the :class:`~membrane.network.cluster.Cluster` constructor.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from membrane.transport.tls import MTLSConfig


@dataclass
class ClusterConfig:
    """Configuration for a Membrane cluster node.

    Attributes:
        node_id: Unique identifier for this node.
        host: Bind address for the HTTP server.
        port: Listen port for the HTTP server.
        peers: Seed peer list as ``"host:port"`` strings.
        heartbeat_interval_sec: Seconds between heartbeats.
        heartbeat_timeout_sec: HTTP timeout for heartbeat
            requests.
        gossip_interval_sec: Seconds between gossip rounds.
        failure_suspect_threshold: Missed heartbeats before
            marking a peer as suspect.
        failure_remove_threshold: Missed heartbeats before
            removing a peer from the membership table.
        max_retries: Max retries for peer HTTP requests.
        retry_delay_sec: Base delay between retries (exponential
            backoff).
        replica_count: Number of replicas per primary fragment.
        enable_gossip: Whether to enable gossip protocol.
        enable_replication: Whether to auto-replicate on store.
        gossip_fanout: Number of peers to gossip with each
            round.
        gossip_max_fragment_entries: Max fragment locations per
            gossip message.
        mtls: Optional
            :class:`~membrane.transport.tls.MTLSConfig`. When set,
            cluster joins and inbound requests must present a
            verified client certificate signed by the cluster's
            CA bundle. ``None`` is supported only for the
            single-node deployment; any multi-node cluster must
            supply this field at 2.0+.
        local_peer_cn: The Common Name this node presents as a
            client cert when calling out to peers. Operators must
            keep this in lock-step with the ``MTLSConfig.allowed_cns``
            allow-list on peers — a peer whose CN is not in the
            list rejects the inbound call.
        default_consistency: Write level applied by
            :func:`op_store` when the incoming fragment's
            ``consistency`` field is missing or matches the
            cluster default. Production clusters leave this at
            ``"strong"`` so every op_store blocks on quorum.
            Tests may override to ``"quorum"`` or ``"eventual"``
            to skip the blocking path.
        quorum_count: Number of replica acks op_store waits for
            under ``strong`` or ``"quorum"`` consistency. Default
            ``2`` matches :attr:`replica_count`; production
            clusters typically set this to
            ``floor(replica_count / 2) + 1``.
        cluster_quorum_timeout_sec: Wall-clock budget for the
            op_store quorum wait. On timeout the write fails
            closed (HTTP 503 + ``Retry-After``); the
            :func:`~membrane.transport.ops.op_store` route
            never silently degrades to a weaker consistency.
        repair_interval_sec: Seconds between anti-entropy
            :meth:`~membrane.replicator.Replicator.repair`
            passes. Default ``60`` keeps production clusters
            continuously converged without flooding the wire.
            Tests and single-node deployments disable this by
            setting the field to a very large value.
    """

    node_id: str = "membrane-0"
    host: str = "0.0.0.0"
    port: int = 8080
    peers: list[str] = field(default_factory=list)
    heartbeat_interval_sec: float = 2.0
    heartbeat_timeout_sec: float = 10.0
    gossip_interval_sec: float = 5.0
    failure_suspect_threshold: int = 2
    failure_remove_threshold: int = 4
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    replica_count: int = 2
    enable_gossip: bool = True
    enable_replication: bool = True
    gossip_fanout: int = 2
    gossip_max_fragment_entries: int = 50
    mtls: "MTLSConfig | None" = None
    local_peer_cn: str = ""
    default_consistency: str = "strong"
    quorum_count: int = 2
    cluster_quorum_timeout_sec: float = 5.0
    repair_interval_sec: float = 60.0
