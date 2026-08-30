"""Cluster configuration for Membrane peer-to-peer networking.

This module defines :class:`ClusterConfig`, the single source of
truth for the runtime parameters that govern a Membrane node's
participation in a cluster: bind addresses, peer seeds, heartbeat
and gossip intervals, failure thresholds, retry policy, and
replication knobs.

Callers typically construct a :class:`ClusterConfig` once at
process start (often loading values from environment variables)
and pass it to the :class:`~membrane.network.cluster.Cluster` constructor.

The v3.0.0 release introduces :func:`validate_config` which
uses :mod:`pydantic` to surface useful diagnostics on
configuration errors (Phase 3.6.5). The
:class:`~membrane.network.cluster.Cluster` constructor calls
the validator before touching state so a misconfigured node
fails at start-up rather than mid-gossip.
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
        lease_timeout_sec: Seconds a peer is considered live
            after its last successful heartbeat. Default
            ``30`` keeps the heartbeat-miss counter redundant
            for production clusters; the heartbeat loop
            refreshes :attr:`~membrane.network.membership.PeerInfo.lease_until`
            to ``now() + lease_timeout_sec`` on every ack.
        gossip_payload_expected_items: Expected number of
            items the Bloom filter will hold, used to size
            ``m_bits`` and ``k_hashes``. Default ``10000`` keeps
            the per-gossip-state payload under 2 KiB for
            single-fragment-per-window workloads and around
            17 KiB for one-million-fragment deployments. The
            actual count overrides the configured value when
            the local Node's fragment set is larger.
        gossip_payload_fpr: Target false-positive rate for the
            gossip Bloom filter. Default ``0.001`` (one in a
            thousand) keeps the precision cost negligible while
            bounding the false-divergence rate.
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
    mtls: "MTLSConfig | None" = None
    local_peer_cn: str = ""
    default_consistency: str = "strong"
    quorum_count: int = 2
    cluster_quorum_timeout_sec: float = 5.0
    repair_interval_sec: float = 60.0
    lease_timeout_sec: float = 30.0
    gossip_payload_expected_items: int = 10_000
    gossip_payload_fpr: float = 0.001
    cross_region_penalty: float = 1.5


def validate_config(
    config: "ClusterConfig | dict",
) -> "ClusterConfig":
    """Validate ``config`` and return a normalized :class:`ClusterConfig`.

    Args:
        config: Either an existing :class:`ClusterConfig` (passed
            through unchanged) or a dict carrying the same
            fields. ``pydantic`` enforces the type / range
            constraints and surfaces a useful diagnostic when
            a value is invalid.

    Returns:
        ClusterConfig: The validated, normalized config.

    Raises:
        ValueError: When validation fails. The message lists
            every offending field so operators can fix multiple
            misconfigurations at once.
    """
    if isinstance(config, ClusterConfig):
        return config
    from pydantic import BaseModel, Field, ValidationError

    class _ConfigModel(BaseModel):
        node_id: str = Field(min_length=1, max_length=128)
        host: str = Field(default="0.0.0.0", min_length=1, max_length=255)
        port: int = Field(default=8080, ge=1, le=65535)
        peers: list[str] = Field(default_factory=list, max_length=4096)
        heartbeat_interval_sec: float = Field(default=2.0, gt=0.0)
        heartbeat_timeout_sec: float = Field(default=10.0, gt=0.0)
        gossip_interval_sec: float = Field(default=5.0, gt=0.0)
        failure_suspect_threshold: int = Field(default=2, ge=1)
        failure_remove_threshold: int = Field(default=4, ge=1)
        max_retries: int = Field(default=3, ge=0)
        retry_delay_sec: float = Field(default=1.0, ge=0.0)
        replica_count: int = Field(default=2, ge=0)
        enable_gossip: bool = True
        enable_replication: bool = True
        gossip_fanout: int = Field(default=2, ge=1)
        gossip_max_fragment_entries: int = Field(default=50, ge=1)
        local_peer_cn: str = ""
        default_consistency: str = Field(default="strong")
        quorum_count: int = Field(default=2, ge=1)
        cluster_quorum_timeout_sec: float = Field(default=5.0, gt=0.0)
        repair_interval_sec: float = Field(default=60.0, gt=0.0)
        lease_timeout_sec: float = Field(default=30.0, gt=0.0)
        gossip_payload_expected_items: int = Field(default=10_000, ge=1)
        gossip_payload_fpr: float = Field(default=0.001, gt=0.0, lt=1.0)
        cross_region_penalty: float = Field(default=1.5, ge=1.0)

    try:
        model = _ConfigModel(**(config or {}))
    except ValidationError as exc:
        msg_lines = ["ClusterConfig validation failed:"]
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            msg_lines.append(f"  - {field}: {err.get('msg')}")
        raise ValueError("\n".join(msg_lines)) from exc
    return ClusterConfig(**model.model_dump())


__all__ = ["ClusterConfig", "validate_config"]
