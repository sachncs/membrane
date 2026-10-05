"""Observability: typed metrics collectors and a Prometheus exposition.

Membrane exposes its runtime state through :class:`MetricsCollector`. Each
subsystem (transport, cluster, persistence) owns its own typed collector;
the aggregate registry is what ``/metrics`` exposes as Prometheus text.

This module deliberately avoids a singleton — the registry is built once at
the composition root (``membrane.server.Server.__init__``) and injected
into each subsystem that needs it.
"""

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

type LabelKey = tuple[str, ...]
INF_LABEL = 'le="+Inf"'


def label_key(names: tuple[str, ...], labels: Mapping[str, str]) -> LabelKey:
    return tuple(str(labels.get(name, "")) for name in names)


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_labels(names: tuple[str, ...], key: LabelKey, extra: str = "") -> str:
    parts = [f'{n}="{escape(v)}"' for n, v in zip(names, key, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


@dataclass
class Counter:
    """A monotonically increasing counter with optional labels.

    Each distinct combination of label values is its own series;
    :attr:`value` is the total across series.
    """

    name: str
    help_text: str
    labels: tuple[str, ...] = ()
    series: dict[LabelKey, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment the series selected by ``labels``.

        Args:
            amount: How much to add. Defaults to 1.
            **labels: Label values keyed by label name.
        """
        key = label_key(self.labels, labels)
        with self.lock:
            self.series[key] = self.series.get(key, 0.0) + amount

    @property
    def value(self) -> float:
        """Total across every series."""
        with self.lock:
            return sum(self.series.values())

    def get(self, **labels: str) -> float:
        """Value of one series."""
        with self.lock:
            return self.series.get(label_key(self.labels, labels), 0.0)


@dataclass
class Gauge:
    """A value that can go up or down, with optional labels."""

    name: str
    help_text: str
    labels: tuple[str, ...] = ()
    series: dict[LabelKey, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float, **labels: str) -> None:
        """Set the series selected by ``labels`` to ``value``."""
        key = label_key(self.labels, labels)
        with self.lock:
            self.series[key] = value

    @property
    def value(self) -> float:
        """The unlabeled value, or the sum across labeled series."""
        with self.lock:
            return sum(self.series.values())

    def get(self, **labels: str) -> float:
        """Value of one series."""
        with self.lock:
            return self.series.get(label_key(self.labels, labels), 0.0)


@dataclass
class HistogramSeries:
    counts: dict[float, int] = field(default_factory=dict)
    total: int = 0
    sum_: float = 0.0


@dataclass
class Histogram:
    """A bucketed histogram of observations, with optional labels."""

    name: str
    help_text: str
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    labels: tuple[str, ...] = ()
    series: dict[LabelKey, HistogramSeries] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, value: float, **labels: str) -> None:
        """Record ``value`` in the series selected by ``labels``."""
        key = label_key(self.labels, labels)
        with self.lock:
            series = self.series.setdefault(key, HistogramSeries())
            series.total += 1
            series.sum_ += value
            for b in self.buckets:
                if value <= b:
                    series.counts[b] = series.counts.get(b, 0) + 1

    @property
    def total(self) -> int:
        """Observation count across every series."""
        with self.lock:
            return sum(s.total for s in self.series.values())


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

    def histogram(
        self,
        name: str,
        help_text: str,
        buckets: tuple[float, ...] | None = None,
        labels: tuple[str, ...] = (),
    ) -> Histogram:
        """Get-or-create a histogram."""
        if name not in self.histograms:
            h = Histogram(name=name, help_text=help_text, labels=labels)
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
        for c in list(self.counters.values()):
            lines.append(f"# HELP {c.name} {c.help_text}")
            lines.append(f"# TYPE {c.name} counter")
            with c.lock:
                series = dict(c.series) or ({} if c.labels else {(): 0.0})
            for key, value in sorted(series.items()):
                lines.append(f"{c.name}{render_labels(c.labels, key)} {value}")
        for g in list(self.gauges.values()):
            lines.append(f"# HELP {g.name} {g.help_text}")
            lines.append(f"# TYPE {g.name} gauge")
            with g.lock:
                series = dict(g.series) or ({} if g.labels else {(): 0.0})
            for key, value in sorted(series.items()):
                lines.append(f"{g.name}{render_labels(g.labels, key)} {value}")
        for h in list(self.histograms.values()):
            lines.append(f"# HELP {h.name} {h.help_text}")
            lines.append(f"# TYPE {h.name} histogram")
            with h.lock:
                hseries = {k: HistogramSeries(dict(v.counts), v.total, v.sum_) for k, v in h.series.items()}
            if not hseries and not h.labels:
                hseries = {(): HistogramSeries()}
            for key, hs in sorted(hseries.items(), key=lambda item: item[0]):
                for b in h.buckets:
                    le = render_labels(h.labels, key, f'le="{b}"')
                    lines.append(f"{h.name}_bucket{le} {hs.counts.get(b, 0)}")
                inf = render_labels(h.labels, key, INF_LABEL)
                lines.append(f"{h.name}_bucket{inf} {hs.total}")
                lines.append(f"{h.name}_sum{render_labels(h.labels, key)} {hs.sum_}")
                lines.append(f"{h.name}_count{render_labels(h.labels, key)} {hs.total}")
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
            labels=("endpoint",),
        )


def default_tenant_metrics() -> TenantMetrics:
    return TenantMetrics()


@dataclass
class ClusterMetrics:
    """Typed collector for cluster membership and replication."""

    registry: MetricsCollector
    tenant: TenantMetrics = field(default_factory=lambda: default_tenant_metrics())

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
class TenantMetrics:
    """Per-tenant namespace metrics.

    Attributes:
        fragment_count: Map from tenant id to the number of
            fragments held locally for that tenant. Updated
            by :meth:`Node.record_tenant_count`.
        operation_count: Map from tenant id to the number of
            store / retrieve / replicate operations performed
            on that tenant's fragments. Updated by
            :meth:`Cluster.record_tenant_op`.
    """

    fragment_count: dict[str, int] = field(default_factory=dict)
    operation_count: dict[str, int] = field(default_factory=dict)

    def bump_fragment(self, tenant_id: str, delta: int = 1) -> None:
        """Update the per-tenant fragment counter.

        Args:
            tenant_id: Tenant id to update.
            delta: Signed increment.
        """
        self.fragment_count[tenant_id] = self.fragment_count.get(tenant_id, 0) + delta

    def bump_operation(self, tenant_id: str, delta: int = 1) -> None:
        """Update the per-tenant operation counter.

        Args:
            tenant_id: Tenant id to update.
            delta: Signed increment.
        """
        self.operation_count[tenant_id] = self.operation_count.get(tenant_id, 0) + delta


@dataclass
class NodeMetrics:
    """Typed collector for per-node fragment store metrics."""

    registry: MetricsCollector
    tenant: TenantMetrics = field(default_factory=TenantMetrics)

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

    @property
    def tenant_fragments(self) -> Gauge:
        return self.registry.gauge(
            "membrane_tenant_fragments",
            "Fragments held locally per tenant.",
            labels=("tenant",),
        )

    def sync_tenant_fragment_gauges(self, counts: Mapping[str, int] | None = None) -> None:
        """Publish per-tenant fragment counts.

        Args:
            counts: Live ``tenant -> fragment count``. Defaults to the
                running totals in :attr:`tenant`.
        """
        current = dict(counts if counts is not None else self.tenant.fragment_count)
        gauge = self.tenant_fragments
        with gauge.lock:
            # Tenants that dropped to zero disappear from the export.
            gauge.series = {(tenant,): float(n) for tenant, n in current.items() if n > 0}


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
