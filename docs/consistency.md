# Consistency levels

Every fragment written with `POST /store` carries a consistency level
(`Fragment.consistency`, part of the v5 wire envelope). A single node
always writes locally; in a cluster the level decides whether the write
also waits for replicas.

| Level | Acknowledged when | On failure |
|-------|-------------------|------------|
| `strong` (default) | `quorum_count` copies exist, the local one included | `503` + `Retry-After: 1`; the local write is rolled back |
| `quorum` | same as `strong` | same as `strong` |
| `eventual` | the local write succeeds | none; replication and gossip converge in the background |

`strong` and `quorum` currently behave identically; `strong` is the
name the wire format uses by default. A fragment sent as `strong` is
written with the node's `--consistency` setting when that is `quorum` or
`eventual`, so operators can relax a cluster without changing clients.

## How a strong write works

1. The node checks the caller's tenant and that the payload bytes are
   in its content store, then writes locally.
2. It sends the fragment to up to `replica_count` healthy peers
   (`POST /replicate`) and waits for `quorum_count - 1` acknowledgements,
   at most `cluster_quorum_timeout_sec`.
3. Enough acks: `200`. Otherwise the local copy is removed (unless it
   already existed before this write) and the caller gets `503` with
   `ack_count` and `required` in the body.

A node with fewer than `quorum_count - 1` healthy peers rejects strong
writes immediately instead of acknowledging a copy it cannot replicate.

## Configuration

| `membrane serve` flag | `ClusterConfig` field | Default | Meaning |
|-----------------------|-----------------------|---------|---------|
| `--consistency` | `default_consistency` | `strong` | Level applied to writes sent as `strong` |
| `--quorum-count` | `quorum_count` | `2` | Copies required, including the local one |
| `--replica-count` | `replica_count` | `2` | Peers a write fans out to |
| — | `cluster_quorum_timeout_sec` | `9.0` | Budget for collecting acks |
| `--heartbeat-interval` | `heartbeat_interval_sec` | `2.0` | Seconds between heartbeats |
| `--failure-remove-threshold` | `failure_remove_threshold` | `4` | Missed heartbeats before a peer is removed |

The quorum timeout (9 s) is deliberately longer than the time it takes
to remove a dead peer (4 × 2 s = 8 s), so a write in flight either
completes or fails with a clear error rather than racing membership
changes.

Sizing rules:

- `quorum_count` must not exceed the number of nodes, or every strong
  write fails. A two-node cluster can use the default of 2; it then
  rejects strong writes while either node is down.
- `quorum_count: 1` makes strong writes local-only.
- Peers authenticate to each other for `/replicate`, so API-key
  clusters need `--peer-api-key-file` with an `admin` key (see
  [Security](security.md)).

## Observability

The node exports Prometheus metrics on `/metrics` (requires the `read`
scope). The ones relevant here:

| Metric | Meaning |
|--------|---------|
| `membrane_requests_total` / `membrane_errors_total` | Per-route request and error counts (a failed strong write is an error on `store`) |
| `membrane_replications_total` / `membrane_replication_failures_total` | Replica pushes and failures |
| `membrane_replication_lag_seconds` | Replication lag per peer |
| `membrane_peers_healthy` / `membrane_peers_total` | Cluster membership |
| `membrane_gossip_failures_total` | Failed gossip rounds |

## See also

- `membrane/transport/ops.py`: `op_store`
- `membrane/quorum.py`: the ack fan-out
- `membrane/network/config.py`: `ClusterConfig`
- [Wire format](wire-format.md)
- [SLOs](operations/slo.md)
