"""Transport metrics instrumentation (Phase 3.2.1).

The v2.0 release constructed :class:`TransportMetrics`,
:class:`ClusterMetrics`, :class:`PersistenceMetrics`, and
:class:`NodeMetrics` in :class:`~membrane.server.Server.__init__`
but never incremented any of the series. The v3.0.0 release
wires the counters into the real call paths so :func:`op_metrics`
produces a populated Prometheus text exposition.

The :func:`record_transport` helper wraps any op result,
records the count + duration, and re-raises any exception as a
recorded error. The v3 ops accept an optional ``metrics``
parameter (the :class:`TransportMetrics` instance).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from membrane.metrics import (
    ClusterMetrics,
    NodeMetrics,
    PersistenceMetrics,
    TransportMetrics,
)

logger = logging.getLogger(__name__)


def record_transport(
    metrics: TransportMetrics | None,
    endpoint: str,
    method: str,
    fn: Callable[[], tuple[int, Any]],
) -> tuple[int, Any]:
    """Run ``fn``, record the result in ``metrics``, and return it.

    Args:
        metrics: Optional :class:`TransportMetrics`. When
            ``None`` the function runs without recording.
        endpoint: The endpoint label, e.g., ``"store"``.
        method: The HTTP method label, e.g., ``"POST"``.
        fn: The op to call. Returns ``(status, body)``.

    Returns:
        tuple[int, Any]: The op's ``(status, body)`` result.
    """
    if metrics is None:
        return fn()
    start = time.monotonic()
    try:
        status, body = fn()
    except Exception as exc:  # pragma: no cover - ops don't raise
        metrics.errors.inc(endpoint=endpoint, exception=type(exc).__name__)
        metrics.duration.observe(time.monotonic() - start)
        raise
    metrics.requests.inc(endpoint=endpoint, method=method, status=str(status))
    metrics.duration.observe(time.monotonic() - start)
    return status, body


def record_persistence(
    metrics: PersistenceMetrics | None,
    kind: str,
    fn: Callable[[], Any],
) -> Any:
    """Run ``fn`` and record the persistence outcome.

    Args:
        metrics: Optional :class:`PersistenceMetrics`.
        kind: The operation kind (e.g., ``"get"``, ``"put"``).
        fn: The op to call.

    Returns:
        Any: The op's return value.
    """
    if metrics is None:
        return fn()
    try:
        result = fn()
    except Exception:  # pragma: no cover
        metrics.operations.inc(kind=kind, outcome="error")
        raise
    metrics.operations.inc(kind=kind, outcome="ok")
    return result


def record_cluster_replication(
    metrics: ClusterMetrics | None,
    success: bool,
) -> None:
    """Record a replication push result on ``metrics``.

    Args:
        metrics: Optional :class:`ClusterMetrics`.
        success: Whether the replication succeeded.
    """
    if metrics is None:
        return
    if success:
        metrics.replications.inc()
    else:
        metrics.replication_failures.inc()


def sync_node_metrics(node: Any, metrics: NodeMetrics) -> None:
    """Refresh the per-node gauges from the live ``node``.

    Args:
        node: The :class:`~membrane.node.Node` to read.
        metrics: The collector whose gauges are updated.
    """
    stats = node.get_stats()
    metrics.fragments.set(float(stats.fragment_count))
    metrics.memory_used_bytes.set(float(stats.memory_used_bytes))
    metrics.memory_limit_bytes.set(float(stats.memory_limit_bytes))
    metrics.tenant.fragment_count = dict(metrics.tenant.fragment_count)
    # The fragment_count gauge total lives on TenantMetrics
    # for finer-grained access; the running aggregate is
    # available via the helpers below.
    metrics.sync_tenant_fragment_gauges()


__all__ = [
    "record_cluster_replication",
    "record_persistence",
    "record_transport",
    "sync_node_metrics",
]
