# SLOs and alerting

Recommended service-level objectives for a Membrane deployment, the Prometheus queries that measure them, and the alerts that protect them. Tune the targets to your workload; the queries and alert structure carry over.

## Before you start

`/metrics` serves Prometheus text format and requires a key with the `read` scope. The Kubernetes `ServiceMonitor` in [`deployment/k8s/`](../../deployment/k8s/) passes one.

| Metric | Labels | Meaning |
|--------|--------|---------|
| `membrane_requests_total` | `endpoint`, `method`, `status` | Requests handled |
| `membrane_errors_total` | `endpoint`, `exception` | Requests that raised an error |
| `membrane_request_duration_seconds` | `endpoint` | Latency histogram |
| `membrane_memory_used_bytes`, `membrane_memory_limit_bytes` | | Node memory |
| `membrane_evictions_total` | `reason` | Evictions: `expired` (TTL) or `capacity` (to make room) |
| `membrane_peers_healthy`, `membrane_peers_total` | | Cluster membership |
| `membrane_replication_lag_seconds` | | Replication lag |
| `membrane_requests_rejected_total` | `reason` | Requests shed by limits |
| `membrane_tls_cert_expiry_seconds` | | Time until the serving certificate expires |
| `membrane_audit_chain_valid` | | 1 while the audit log's hash chain verifies |

`endpoint` is the operation name: `store`, `retrieve`, `prefill`, `replicate`, `inventory`, `heartbeat`, `gossip`, and so on.

## Objectives

### Latency

| Operation | Target |
|-----------|--------|
| `retrieve` | p99 < 50 ms |
| `store` (`eventual`) | p99 < 100 ms |
| `store` (`strong`) | p99 < 500 ms, including one replica round trip |

```text title="PromQL: retrieve p99"
histogram_quantile(0.99,
  sum by (le) (rate(membrane_request_duration_seconds_bucket{endpoint="retrieve"}[5m])))
```

### Availability

Target: 99.9% of requests over 30 days do not return a 5xx status.

```text title="PromQL: 30-day availability"
1 - (
  sum(rate(membrane_requests_total{status=~"5.."}[30d]))
  / sum(rate(membrane_requests_total[30d]))
)
```

A `503` from a `strong` write that could not reach quorum counts against this budget: it means the cluster lacked healthy replicas.

### Capacity

A full node is normal: `store` evicts expired fragments, then the least valuable ones, to make room, and `/readyz` stays ready. Watch the mix of evictions instead:

```text title="PromQL: evictions by reason"
sum by (reason) (rate(membrane_evictions_total[15m]))
```

Sustained `capacity` evictions of fragments that are later requested again mean the node is undersized. See [Capacity planning](capacity.md).

## Alerts

### Error-budget burn rate

For a 99.9% objective (an error budget of 0.1%), use multi-window burn-rate alerts:

| Severity | Condition |
|----------|-----------|
| Page | 5xx ratio above 14.4 × 0.1% over 1 hour **and** over 5 minutes |
| Page | 5xx ratio above 6 × 0.1% over 6 hours **and** over 30 minutes |
| Ticket | 5xx ratio above 1 × 0.1% over 3 days |

### Cluster and security

| Severity | Condition | Why |
|----------|-----------|-----|
| Page | `membrane_peers_healthy < membrane_peers_total` for more than 2 minutes | A peer is down; strong writes may start failing |
| Page | `membrane_audit_chain_valid == 0` | The audit log was edited, truncated, or reordered. See [Incident response](incident-response.md) |
| Ticket | `membrane_tls_cert_expiry_seconds < 14 * 86400` | The serving certificate expires within 14 days |
| Ticket | `rate(membrane_requests_rejected_total[15m]) > 0` sustained | Clients are being shed by concurrency or rate limits |

## References

- Google SRE Workbook, [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/)
