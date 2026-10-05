# SLOs and Error Budget

Suggested service-level objectives for a Membrane deployment, and the
Prometheus queries that measure them. Tune the targets to your workload;
the queries are what matter.

`/metrics` serves Prometheus text and requires a key with the `read`
scope (the Kubernetes `ServiceMonitor` passes one).

## Metrics used

| Metric | Labels | Meaning |
|--------|--------|---------|
| `membrane_requests_total` | `endpoint`, `method`, `status` | Requests handled |
| `membrane_errors_total` | `endpoint`, `exception` | Requests that raised |
| `membrane_request_duration_seconds` | `endpoint` | Latency histogram |
| `membrane_memory_used_bytes` / `membrane_memory_limit_bytes` | | Node memory |
| `membrane_evictions_total` | `reason` | Evictions (`expired`, `lru`, …) |
| `membrane_peers_healthy` / `membrane_peers_total` | | Cluster membership |
| `membrane_replication_lag_seconds` | | Replication lag |

`endpoint` is the operation name: `store`, `retrieve`, `prefill`,
`replicate`, `inventory`, `heartbeat`, `gossip`, and so on.

## Latency

| Operation | Target |
|-----------|--------|
| `retrieve` | p99 < 50 ms |
| `store` (`eventual`) | p99 < 100 ms |
| `store` (`strong`) | p99 < 500 ms (includes one replica round trip) |

```promql
histogram_quantile(0.99,
  sum by (le) (rate(membrane_request_duration_seconds_bucket{endpoint="retrieve"}[5m])))
```

## Availability

Target: 99.9 % of requests over 30 days do not return 5xx.

```promql
1 - (
  sum(rate(membrane_requests_total{status=~"5.."}[30d]))
  / sum(rate(membrane_requests_total[30d]))
)
```

A `503` from a `strong` write that could not reach quorum counts against
this budget: it means the cluster lacked healthy replicas.

## Capacity

A full node is normal: `store` evicts expired and then least-valuable
fragments to make room, and `/readyz` stays ready. Watch the eviction
mix instead:

```promql
sum by (reason) (rate(membrane_evictions_total[15m]))
```

Sustained `lru` evictions of fragments that are later requested again
mean the node is undersized (see [Capacity planning](capacity.md)).

## Burn-rate alerts

For a 99.9 % objective (error budget 0.1 %):

| Severity | Condition |
|----------|-----------|
| Page | 5xx ratio > 14.4 × 0.1 % over 1 h **and** over 5 m |
| Page | 5xx ratio > 6 × 0.1 % over 6 h **and** over 30 m |
| Ticket | 5xx ratio > 1 × 0.1 % over 3 d |

Also alert when `membrane_peers_healthy < membrane_peers_total` for
more than 2 minutes.

## References

- Google SRE workbook, *Alerting on SLOs*: https://sre.google/workbook/alerting-on-slos/
