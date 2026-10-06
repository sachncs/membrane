"""Cluster configuration for Membrane peer-to-peer networking.

:class:`ClusterConfig` holds the runtime parameters that govern a node's
participation in a cluster: bind addresses, peer seeds, heartbeat and
gossip intervals, failure thresholds, retry policy, and replication
knobs. ``membrane serve`` derives it from
:class:`~membrane.runtime.settings.ServerSettings`
(:meth:`ServerSettings.cluster_config`); library users construct it
directly.

Every construction is validated: :meth:`ClusterConfig.__post_init__`
checks each field against :data:`CONSTRAINTS` and raises one
``ValueError`` listing every problem, so a misconfigured node fails at
start-up rather than mid-gossip.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from membrane.transport.tls import MTLSConfig


CONSISTENCY_LEVELS = frozenset({"strong", "quorum", "eventual"})

#: Field -> (check, description). A field fails when ``check(value)`` is False.
CONSTRAINTS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "node_id": (lambda v: 1 <= len(v) <= 128, "must be 1-128 characters"),
    "host": (lambda v: 1 <= len(v) <= 255, "must be 1-255 characters"),
    "port": (lambda v: 0 <= v <= 65535, "must be 0-65535 (0 picks a free port)"),
    "peers": (lambda v: len(v) <= 4096, "at most 4096 seeds"),
    "heartbeat_interval_sec": (lambda v: v > 0, "must be > 0"),
    "heartbeat_timeout_sec": (lambda v: v > 0, "must be > 0"),
    "gossip_interval_sec": (lambda v: v > 0, "must be > 0"),
    "failure_suspect_threshold": (lambda v: v >= 1, "must be >= 1"),
    "failure_remove_threshold": (lambda v: v >= 1, "must be >= 1"),
    "max_retries": (lambda v: v >= 0, "must be >= 0"),
    "retry_delay_sec": (lambda v: v >= 0, "must be >= 0"),
    "replica_count": (lambda v: v >= 0, "must be >= 0"),
    "gossip_fanout": (lambda v: v >= 1, "must be >= 1"),
    "max_location_entries": (lambda v: v >= 1, "must be >= 1"),
    "gossip_max_fragment_entries": (lambda v: v >= 1, "must be >= 1"),
    "advertise_host": (lambda v: len(v) <= 255, "at most 255 characters"),
    "default_consistency": (lambda v: v in CONSISTENCY_LEVELS, "must be strong, quorum, or eventual"),
    "quorum_count": (lambda v: v >= 1, "must be >= 1"),
    "cluster_quorum_timeout_sec": (lambda v: v > 0, "must be > 0"),
    "repair_interval_sec": (lambda v: v > 0, "must be > 0"),
    "lease_timeout_sec": (lambda v: v > 0, "must be > 0"),
    "cross_region_penalty": (lambda v: v >= 1, "must be >= 1"),
}


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
        max_location_entries: Fragment hashes whose locations the
            registry keeps (least recently recorded forgotten first).
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
        advertise_host: Host peers use to reach this node. Empty
            means the bind host, or the machine FQDN when binding
            to a wildcard address.
        default_consistency: Write level applied by
            :func:`op_store` when the incoming fragment's
            ``consistency`` field is missing or matches the
            cluster default. Production clusters leave this at
            ``"strong"`` so every op_store blocks on quorum.
            Tests may override to ``"quorum"`` or ``"eventual"``
            to skip the blocking path.
        quorum_count: Number of copies (the local write included)
            that must exist before op_store acknowledges a
            ``strong`` or ``"quorum"`` write; op_store waits for
            ``quorum_count - 1`` peer acks out of a fan-out to
            :attr:`replica_count` peers. Default ``2`` (local plus
            one replica). ``1`` makes strong writes local-only.
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
        lease_timeout_sec: Seconds a peer is considered live
            after its last successful heartbeat. Default
            ``30`` keeps the heartbeat-miss counter redundant
            for production clusters; the heartbeat loop
            refreshes :attr:`~membrane.network.membership.PeerInfo.lease_until`
            to ``now() + lease_timeout_sec`` on every ack.
        cross_region_penalty: Multiplier applied when
            :class:`~membrane.shard.Shard`'s
            :meth:`locality_scored_assign` ranks a cross-region
            candidate above a same-region one. ``1.0`` disables
            the preference (pure bandwidth ranking); higher
            values tighten the cross-region preference. Default
            ``1.5`` matches the design plan.
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
    max_location_entries: int = 200_000
    mtls: MTLSConfig | None = None
    local_peer_cn: str = ""
    advertise_host: str = ""
    default_consistency: str = "strong"
    quorum_count: int = 2
    # Default ``9.0`` (was ``5.0``) so the cluster_quorum_timeout
    # is **strictly greater** than
    # ``failure_remove_threshold * heartbeat_interval_sec``
    # (= 4 * 2 = 8 s at the documented defaults). A peer is
    # therefore held around long enough for the quorum write
    # to either succeed or fail-closed with a typed error,
    # rather than being removed from the membership while the
    # quorum write is still in flight.
    cluster_quorum_timeout_sec: float = 9.0
    repair_interval_sec: float = 60.0
    lease_timeout_sec: float = 30.0
    cross_region_penalty: float = 1.5

    def __post_init__(self) -> None:
        """Validate every field against :data:`CONSTRAINTS`.

        Raises:
            ValueError: Listing every field that fails its constraint.
        """
        problems = [
            f"  - {name}: {description} (got {getattr(self, name)!r})"
            for name, (check, description) in CONSTRAINTS.items()
            if not check(getattr(self, name))
        ]
        if problems:
            raise ValueError("ClusterConfig validation failed:\n" + "\n".join(problems))


def validate_config(config: ClusterConfig | dict) -> ClusterConfig:
    """Return ``config`` as a validated :class:`ClusterConfig`.

    Args:
        config: An existing :class:`ClusterConfig` (already validated at
            construction) or a dict of its fields.

    Returns:
        ClusterConfig: The validated config.

    Raises:
        ValueError: When a field is unknown or out of range; the message
            lists every offending field.
    """
    if isinstance(config, ClusterConfig):
        return config
    unknown = sorted(set(config or {}) - set(ClusterConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"ClusterConfig validation failed:\n  - unknown fields: {', '.join(unknown)}")
    return ClusterConfig(**(config or {}))


__all__ = ["CONSISTENCY_LEVELS", "CONSTRAINTS", "ClusterConfig", "validate_config"]
