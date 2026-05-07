"""Cluster configuration for Membrane peer-to-peer networking."""

from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    """Configuration for a Membrane cluster node.

    Attributes:
        node_id: Unique identifier for this node.
        host: Bind address for the HTTP server.
        port: Listen port for the HTTP server.
        peers: Seed peer list as "host:port" strings.
        heartbeat_interval_sec: Seconds between heartbeats.
        heartbeat_timeout_sec: HTTP timeout for heartbeat requests.
        gossip_interval_sec: Seconds between gossip rounds.
        failure_suspect_threshold: Missed heartbeats before marking suspect.
        failure_remove_threshold: Missed heartbeats before removing peer.
        max_retries: Max retries for peer HTTP requests.
        retry_delay_sec: Base delay between retries (exponential backoff).
        replica_count: Number of replicas per primary fragment.
        enable_gossip: Whether to enable gossip protocol.
        enable_replication: Whether to auto-replicate on store.
        gossip_fanout: Number of peers to gossip with each round.
        gossip_max_fragment_entries: Max fragment locations per gossip message.
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
