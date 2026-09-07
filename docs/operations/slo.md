# SLOs and Error Budget

This document defines Membrane's service-level objectives (SLOs) and the
error-budget policy that follows from them. These targets are based on
the throughput model in `membrane/model/throughput.py` and the
observability surfaced via `/metrics` (Prometheus text exposition).

## Latency SLOs

| Operation | Target | Measurement |
|---|---|---|
| `GET /retrieve` (cache hit) | p50 < 10 ms, p99 < 50 ms | histogram `membrane_request_duration_seconds{endpoint="/retrieve"}` |
| `POST /store` (write through) | p50 < 25 ms, p99 < 100 ms | histogram `membrane_request_duration_seconds{endpoint="/store"}` |
| `POST /prefill` (compute) | p50 < 500 ms, p99 < 2 s | histogram `membrane_request_duration_seconds{endpoint="/prefill"}` |
| `GET /livez` | p99 < 5 ms | histogram `membrane_request_duration_seconds{endpoint="/livez"}` |

## Availability SLO

* **Target**: 99.9 % of `/retrieve` requests return < 500 ms over a rolling
  30-day window.
* **Error budget**: 43.2 minutes of downtime per 30-day window.

## Error-rate SLO

* **Target**: < 0.1 % of requests return 5xx over a rolling 30-day window.
* **Error budget**: 2,160 failed requests per million.

## Capacity SLO

* **Memory**: `membrane_memory_used_bytes / membrane_memory_limit_bytes < 0.9`
* **Fragment count**: bounded by `max_count` config; `/readyz` returns 503
  when capacity is exhausted.

## Burn-rate alerts

| Severity | Condition |
|---|---|
| Page | Error budget burning at > 14.4x for 1 h (consumes 2 % per hour) |
| Page | Error budget burning at > 6x for 6 h |
| Ticket | Error budget burning at > 1x for 24 h |

## Reporting

The SLO dashboard consumes `/metrics` and `/metrics.json` directly. No
external reporting tool is required.

## References

* Google SRE workbook: https://sre.google/workbook/table-contents/
* Prometheus SLO recording rules: https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/#recording-rules
