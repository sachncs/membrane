"""Per-peer replication-lag gauge (Phase 3.2.2).

The v2.0 release stamped ``PeerInfo.last_heartbeat`` at every
successful heartbeat round but never surfaced the lag in a
metric. The v3.0.0 release exposes a per-peer gauge so an
operator watching :func:`op_heartbeat` can see which nodes have
stopped responding.

The lag is computed as ``now - last_heartbeat``. A node that
never beat is reported as ``inf`` and surfaced as ``-1`` in
the JSON fallback (the Prometheus convention uses positive
numerics).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from membrane.metrics import MetricsCollector
from membrane.network.membership import Membership

REPLICATION_LAG_GAUGE: str = "membrane_replication_lag_seconds"
"""Per-peer gauge. The label is the peer node id."""


@dataclass(frozen=True)
class PeerLagSnapshot:
    """Result of :func:`snapshot_peer_lag`.

    Attributes:
        lag_seconds: Map from peer node id to the seconds since
            the last successful heartbeat. Missing peers
            (never beat) are reported as ``math.inf``.
        now: Monotonic clock used for the computation.
    """

    lag_seconds: dict[str, float]
    now: float


def snapshot_peer_lag(membership: Membership, *, now: float | None = None) -> PeerLagSnapshot:
    """Compute the replication lag for every known peer.

    Args:
        membership: The cluster membership table.
        now: Optional monotonic clock override. ``None`` reads
            ``time.monotonic()`` so the helper is deterministic
            in tests when an explicit ``now`` is supplied.

    Returns:
        PeerLagSnapshot: Per-peer ``now - last_heartbeat``.
    """
    current = time.monotonic() if now is None else now
    lag_seconds: dict[str, float] = {}
    for peer in membership.snapshot():
        last = peer.last_heartbeat
        if last <= 0.0:
            lag_seconds[peer.node_id] = math.inf
        else:
            lag_seconds[peer.node_id] = max(0.0, current - last)
    return PeerLagSnapshot(lag_seconds=lag_seconds, now=current)


def render_prometheus_gauge(snapshot: PeerLagSnapshot, gauge_name: str = REPLICATION_LAG_GAUGE) -> str:
    """Render :class:`PeerLagSnapshot` as a Prometheus text snippet.

    Args:
        snapshot: The lag snapshot to render.
        gauge_name: Gauge metric name.

    Returns:
        str: Lines in Prometheus text exposition format
        (`# HELP`, `# TYPE`, then one line per peer).
    """
    lines = [
        f"# HELP {gauge_name} Seconds since the most recent successful heartbeat from each peer.",
        f"# TYPE {gauge_name} gauge",
    ]
    for peer_id, lag in sorted(snapshot.lag_seconds.items()):
        # The Prometheus convention uses positive numerics; an
        # "infinite" lag surfaces as a large sentinel value (10
        # years) so dashboards alert on the threshold without a
        # +Inf special case.
        value = lag if math.isfinite(lag) else 10 * 365 * 24 * 3600.0
        lines.append(f'{gauge_name}{{peer="{peer_id}"}} {value:g}')
    if not snapshot.lag_seconds:
        lines.append(f"{gauge_name} 0")
    return "\n".join(lines) + "\n"


def record_replication_lag(
    registry: MetricsCollector,
    membership: Membership,
) -> PeerLagSnapshot:
    """Record the per-peer lag gauges into ``registry``.

        Args:
            registry: The Prometheus registry. The helper
                creates (or reuses) a Gauge per peer id and
                stamps the latest value.
            membership: Cluster membership table.

        Returns:
            PeerLagSnapshot: The snapshot that was recorded.

        Note:
            The :class:`MetricsCollector` primitive is scalar
            today; this helper stores the per-peer map under
            :attr:`MetricsCollector.gauges` keyed by
            ``gauge_name + ':' + peer_id`` and renders the
            per-peer lines via :func:`render_prometheus_gauge`.
        """
    snapshot = snapshot_peer_lag(membership)
    for peer_id, lag in snapshot.lag_seconds.items():
        gauge = registry.gauge(
            f"{REPLICATION_LAG_GAUGE}:{peer_id}",
            "Per-peer replication lag seconds.",
        )
        gauge.set(lag if math.isfinite(lag) else 10 * 365 * 24 * 3600.0)
    return snapshot


__all__ = [
    "REPLICATION_LAG_GAUGE",
    "PeerLagSnapshot",
    "record_replication_lag",
    "render_prometheus_gauge",
    "snapshot_peer_lag",
]
