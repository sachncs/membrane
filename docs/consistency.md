# Consistency levels

Choose how much durability each write gets. A single node always writes locally; in a cluster, the consistency level decides whether a write also waits for replicas before it is acknowledged. The default, `strong`, survives the loss of the node that took the write.

## Choosing a level

| Level | Acknowledged when | On failure | Use when |
|-------|-------------------|------------|----------|
| `strong` (default) | `quorum_count` copies exist, the local one included | `503` with `Retry-After: 1`; the local write is rolled back | A cached prefix must survive a node failure |
| `quorum` | Same as `strong` | Same as `strong` | You prefer the explicit name |
| `eventual` | The local write succeeds | None; replication converges in the background | Throughput matters more than surviving an immediate failure |

`strong` and `quorum` behave identically today; `strong` is the name the wire format uses by default.

Every fragment carries its level (part of the [wire format](wire-format.md)). A fragment sent as `strong` is written with the node's `--consistency` setting when that is `quorum` or `eventual`, so operators can relax a cluster without changing clients.

## How a strong write works

1. **Local write.** The node checks the caller's tenant and that the payload bytes are in its content store, then writes locally.
2. **Fan-out.** It sends the fragment to up to `replica_count` healthy peers in parallel: the bytes first (`PUT /blobs/{payload_ref}`, verified by SHA-256 on the peer), then the metadata (`POST /replicate`).
3. **Acknowledgement.** It waits up to `cluster_quorum_timeout_sec` for `quorum_count - 1` peers to acknowledge. A peer acknowledges only when it holds both the bytes and the metadata, so an acknowledged write survives the loss of the node that took it.
4. **Result.** With enough acknowledgements the caller gets `200`. Otherwise the local copy is removed (unless it existed before this write), and the caller gets `503` with `ack_count` and `required` in the body.

> [!NOTE]
> A node with fewer than `quorum_count - 1` healthy peers rejects strong writes immediately, rather than acknowledging a copy it cannot replicate.

## Ownership and rebalancing

The node that takes a write becomes the fragment's **primary**. A background replicator keeps every primary's replicas complete: it checks new primaries on every sweep, and runs a digest-based full pass every `repair_interval_sec`.

Each node is on the same consistent-hash ring as its peers, so every node agrees on where each fragment belongs. When the set of healthy nodes changes, each node hands every primary that the ring now assigns elsewhere to its new owner, at most 50 per second, and primaries settle on their owners. A draining node (`SIGTERM`) hands off all its primaries the same way.

A hand-off counts only after the new owner's copy is verified (metadata present and payload digest equal), so a failed hand-off leaves ownership where it was.

## Configuration

| `membrane serve` flag | `ClusterConfig` field | Default | Meaning |
|-----------------------|-----------------------|---------|---------|
| `--consistency` | `default_consistency` | `strong` | Level applied to writes sent as `strong` |
| `--quorum-count` | `quorum_count` | `2` | Copies required, including the local one |
| `--replica-count` | `replica_count` | `2` | Peers a write fans out to |
| — | `cluster_quorum_timeout_sec` | `9.0` | Budget for collecting acknowledgements |
| `--heartbeat-interval` | `heartbeat_interval_sec` | `2.0` | Seconds between heartbeats |
| `--failure-remove-threshold` | `failure_remove_threshold` | `4` | Missed heartbeats before a peer is removed |

The quorum timeout (9 seconds) is deliberately longer than the time it takes to remove a dead peer (4 × 2 seconds), so a write in flight either completes or fails with a clear error rather than racing a membership change.

### Sizing

- `quorum_count` must not exceed the number of nodes, or every strong write fails.
- A two-node cluster can keep the default of 2, but it then rejects strong writes while either node is down.
- `quorum_count: 1` makes strong writes local-only.
- Peers authenticate to each other for `/replicate`, so API-key clusters need `--peer-api-key-file` with an `admin` key. See [Security](security.md).

## Observability

These Prometheus metrics, served on `/metrics` with the `read` scope, show replication health:

| Metric | Meaning |
|--------|---------|
| `membrane_requests_total`, `membrane_errors_total` | Requests and errors per route; a failed strong write is an error on `store` |
| `membrane_replications_total`, `membrane_replication_failures_total` | Replica pushes and failures |
| `membrane_replication_lag_seconds` | Replication lag per peer |
| `membrane_peers_healthy`, `membrane_peers_total` | Cluster membership |
| `membrane_gossip_failures_total` | Failed gossip rounds |

[SLOs](operations/slo.md) turns these into alerts.

## Implementation reference

- `membrane/transport/ops.py`: `op_store`, the write path
- `membrane/quorum.py`: the acknowledgement fan-out
- `membrane/replicator.py`: repair and rebalancing
- `membrane/network/config.py`: `ClusterConfig`
