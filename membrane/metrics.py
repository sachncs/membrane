"""Observability: typed metrics collectors and a Prometheus exposition.

Membrane exposes its runtime state through :class:`MetricsCollector`. Each
subsystem (transport, cluster, persistence) owns its own typed collector;
the aggregate registry is what ``/metrics`` exposes as Prometheus text.

This module deliberately avoids a singleton — the registry is built once at
the composition root (``membrane.server.Server.__init__``) and injected
into each subsystem that needs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class Counter:
    """A monotonically increasing counter."""

    name: str
    help_text: str
    labels: tuple[str, ...] = ()
    value: float = 0.0

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment the counter.

        Args:
            amount: How much to add. Defaults to 1.
            **labels: Label values keyed by label name.
        """
        self.value += amount


@dataclass
class Gauge:
    """A value that can go up or down."""

    name: str
    help_text: str
    labels: tuple[str, ...] = ()
    value: float = 0.0

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge to ``value``."""
        self.value = value


@dataclass
class Histogram:
    """A bucketed histogram of observations."""

    name: str
    help_text: str
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    counts: dict[float, int] = field(default_factory=dict)
    total: int = 0
    sum_: float = 0.0

    def observe(self, value: float) -> None:
        """Record ``value`` in the histogram."""
        self.total += 1
        self.sum_ += value
        for b in self.buckets:
            if value <= b:
                self.counts[b] = self.counts.get(b, 0) + 1


class MetricsCollector:
    """Aggregate registry of counters, gauges, and histograms.

    Each subsystem owns its own typed collector (e.g., ``NodeMetrics``);
    they all register their series with a shared ``MetricsCollector`` that
    is constructed at the composition root and passed in.
    """

    def __init__(self) -> None:
        self.counters: dict[str, Counter] = {}
        self.gauges: dict[str, Gauge] = {}
        self.histograms: dict[str, Histogram] = {}

    def counter(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> Counter:
        """Get-or-create a counter."""
        if name not in self.counters:
            self.counters[name] = Counter(name=name, help_text=help_text, labels=labels)
        return self.counters[name]

    def gauge(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> Gauge:
        """Get-or-create a gauge."""
        if name not in self.gauges:
            self.gauges[name] = Gauge(name=name, help_text=help_text, labels=labels)
        return self.gauges[name]

    def histogram(self, name: str, help_text: str, buckets: tuple[float, ...] | None = None) -> Histogram:
        """Get-or-create a histogram."""
        if name not in self.histograms:
            h = Histogram(name=name, help_text=help_text)
            if buckets is not None:
                h.buckets = buckets
            self.histograms[name] = h
        return self.histograms[name]

    def render(self) -> str:
        """Render the registry as Prometheus text exposition.

        Returns:
            str: A string in the Prometheus text exposition format
            (``Content-Type: text/plain; version=0.0.4``).
        """
        lines: list[str] = []
        for c in self.counters.values():
            lines.append(f"# HELP {c.name} {c.help_text}")
            lines.append(f"# TYPE {c.name} counter")
            lines.append(f"{c.name} {c.value}")
        for g in self.gauges.values():
            lines.append(f"# HELP {g.name} {g.help_text}")
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{g.name} {g.value}")
        for h in self.histograms.values():
            lines.append(f"# HELP {h.name} {h.help_text}")
            lines.append(f"# TYPE {h.name} histogram")
            cumulative = 0
            for b in h.buckets:
                cumulative = h.counts.get(b, 0)
                lines.append(f'{h.name}_bucket{{le="{b}"}} {cumulative}')
            lines.append(f'{h.name}_bucket{{le="+Inf"}} {h.total}')
            lines.append(f"{h.name}_sum {h.sum_}")
            lines.append(f"{h.name}_count {h.total}")
        return "\n".join(lines) + "\n"


@dataclass
class TransportMetrics:
    """Typed collector for transport-layer metrics."""

    registry: MetricsCollector

    @property
    def requests(self) -> Counter:
        return self.registry.counter(
            "membrane_requests_total",
            "Total inbound HTTP/gRPC requests by endpoint, method, and status.",
            labels=("endpoint", "method", "status"),
        )

    @property
    def errors(self) -> Counter:
        return self.registry.counter(
            "membrane_errors_total",
            "Total request errors by endpoint and exception class.",
            labels=("endpoint", "exception"),
        )

    @property
    def duration(self) -> Histogram:
        return self.registry.histogram(
            "membrane_request_duration_seconds",
            "End-to-end request duration by endpoint.",
        )


@dataclass
class ClusterMetrics:
    """Typed collector for cluster membership and replication."""

    registry: MetricsCollector

    @property
    def peers_total(self) -> Gauge:
        return self.registry.gauge("membrane_peers_total", "Total peers known to this node.")

    @property
    def peers_healthy(self) -> Gauge:
        return self.registry.gauge("membrane_peers_healthy", "Healthy peers.")

    @property
    def gossip_rounds(self) -> Counter:
        return self.registry.counter("membrane_gossip_rounds_total", "Completed gossip rounds.")

    @property
    def gossip_failures(self) -> Counter:
        return self.registry.counter("membrane_gossip_failures_total", "Failed gossip sends.")

    @property
    def replications(self) -> Counter:
        return self.registry.counter("membrane_replications_total", "Replicated fragment pushes.")

    @property
    def replication_failures(self) -> Counter:
        return self.registry.counter("membrane_replication_failures_total", "Failed replication pushes.")


@dataclass
class PersistenceMetrics:
    """Typed collector for persistence backend metrics."""

    registry: MetricsCollector

    @property
    def operations(self) -> Counter:
        return self.registry.counter(
            "membrane_persistence_operations_total",
            "Persistence operations by kind and outcome.",
            labels=("kind", "outcome"),
        )

    @property
    def circuit_open(self) -> Gauge:
        return self.registry.gauge(
            "membrane_persistence_circuit_open", "1 when the persistence circuit breaker is open."
        )


@dataclass
class NodeMetrics:
    """Typed collector for per-node fragment store metrics."""

    registry: MetricsCollector

    @property
    def fragments(self) -> Gauge:
        return self.registry.gauge("membrane_fragments_total", "Total fragments held locally.")

    @property
    def memory_used_bytes(self) -> Gauge:
        return self.registry.gauge("membrane_memory_used_bytes", "Memory used by local fragment store.")

    @property
    def memory_limit_bytes(self) -> Gauge:
        return self.registry.gauge("membrane_memory_limit_bytes", "Configured memory budget.")

    @property
    def evictions(self) -> Counter:
        return self.registry.counter(
            "membrane_evictions_total",
            "Evicted fragments by reason (expired, lru, capacity, graph).",
            labels=("reason",),
        )


def metrics_summary(registry: MetricsCollector) -> Mapping[str, float]:
    """Return a flat ``name -> value`` summary (counters and gauges only)."""
    return {
        **{c.name: c.value for c in registry.counters.values()},
        **{g.name: g.value for g in registry.gauges.values()},
    }


__all__ = [
    "ClusterMetrics",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsCollector",
    "NodeMetrics",
    "PersistenceMetrics",
    "TransportMetrics",
    "metrics_summary",
]
