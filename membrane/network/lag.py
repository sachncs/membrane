"""Per-peer replication-lag gauge.

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
        now: Wall-clock time used for the computation.
    """

    lag_seconds: dict[str, float]
    now: float


def snapshot_peer_lag(membership: Membership, *, now: float | None = None) -> PeerLagSnapshot:
    """Compute the replication lag for every known peer.

    Args:
        membership: The cluster membership table.
        now: Optional wall-clock override (``time.time()`` scale, like
            ``PeerInfo.last_heartbeat``); ``None`` reads the clock.

    Returns:
        PeerLagSnapshot: Per-peer ``now - last_heartbeat``.
    """
    # ``last_heartbeat`` is wall-clock time (``time.time()``).
    current = time.time() if now is None else now
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
        one gauge labeled by ``peer``.
    """
    snapshot = snapshot_peer_lag(membership)
    gauge = registry.gauge(REPLICATION_LAG_GAUGE, "Seconds since the last heartbeat from each peer.", labels=("peer",))
    for peer_id, lag in snapshot.lag_seconds.items():
        gauge.set(lag if math.isfinite(lag) else 10 * 365 * 24 * 3600.0, peer=peer_id)
    return snapshot


__all__ = [
    "REPLICATION_LAG_GAUGE",
    "PeerLagSnapshot",
    "record_replication_lag",
    "render_prometheus_gauge",
    "snapshot_peer_lag",
]
