# Incident Response Playbook

Steps for the on-call engineer when a Membrane alert fires. Pair with
[SLOs](slo.md) for the alert definitions and
[Backup & restore](backup-restore.md) for recovery.

## Severity levels

| Level | Definition | Response time |
|-------|------------|---------------|
| SEV-1 | Cluster unavailable, or data exposed across tenants | < 15 min |
| SEV-2 | Error budget burning fast; strong writes failing | < 1 h |
| SEV-3 | Degradation without budget impact | < 4 h |

## Triage

All routes except the probes need a key; use one with `read` scope
(`admin` for `/admin/*`).

```bash
H="Authorization: Bearer $MEMBRANE_API_KEY"
curl -fsS https://membrane.example/livez                     # process up?
curl -fsS https://membrane.example/readyz                    # serving?
curl -fsS -H "$H" https://membrane.example/peers             # cluster view
curl -fsS -H "$H" https://membrane.example/metrics | grep -v '^#'
```

Or from a workstation: `MEMBRANE_API_KEY=… membrane cluster-status --host <node>`.

`/readyz` returns `503` while the node drains (`{"status": "draining"}`)
or when the node object is missing; a full node is ready. If `/livez`
fails, the process is down: check `kubectl get pods`, `kubectl logs`,
or `journalctl -u membrane`.

Logs are JSON in containers (`MEMBRANE_LOG_FORMAT=json`). Every line
written while handling a request carries its `request_id`, which is
also returned to the caller in `X-Request-ID`. Ask the caller for it
and filter on it:

```bash
kubectl logs membrane-0 | jq 'select(.request_id == "0199...")'
```

### Traces

Start nodes with `--otel-endpoint http://collector:4317` (or set
`OTEL_EXPORTER_OTLP_ENDPOINT`; install `membrane[otel]`). Each request
is a server span tagged with its `membrane.request_id`; strong writes
add `quorum.replicate` and per-replica `replication.push` spans, and
peer calls carry `traceparent`, so one write shows up as a single
trace across the nodes it touched. Search by request ID to go from a
client error to the trace.

### Live debugging (Python 3.14)

Without restarting the process, on a host or container with ptrace
permission (add `--cap-add SYS_PTRACE` for Docker):

```bash
python -m asyncio ps <pid>     # tasks of the uvicorn event loop
python -m pdb -p <pid>         # attach a debugger to the live process (PEP 768)
```

Stuck background work shows as a thread name: `membrane-http`,
`membrane-gossip`, `membrane-replication`, `membrane-heartbeat`,
`membrane-persistence`, `membrane-checkpoint`, `membrane-sweeper`. A
loop that raises is logged (`cluster <name> loop crashed; restarting
it`) and restarted. Any other thread that dies is logged at CRITICAL by
the `membrane.threads` logger.

## Common scenarios

### Strong writes return 503 (`quorum not met`)

The node could not get `quorum_count - 1` peer acknowledgements.

1. `GET /peers`: are peers healthy? Compare `membrane_peers_healthy`
   with `membrane_peers_total`.
2. Peer calls failing with `401` / `403` in the logs: the peer API key
   is missing, wrong, or lacks `admin`.
3. `rejected by SSRF policy` in the logs: the peer's address is outside
   `MEMBRANE_PEER_NETWORKS`.
4. To keep accepting writes during the incident, temporarily set
   `MEMBRANE_CONSISTENCY=eventual` and roll the pods.

### Node will not start

The process exits with code 2 and logs the reason. Common ones:

- `Refusing to serve unauthenticated`: no API keyfile or TLS configured.
- `cannot read API keyfile` / `contains no valid keys`: secret not
  mounted, or empty.
- `has mode 0644; other users can read or change it`: a secret file
  (keyfile, TLS key, data key) is readable by other users. Run
  `chmod 600` (Kubernetes: `defaultMode: 0440` with `fsGroup`).
- `unknown compute backend` / `unknown content store`: a typo, or the
  plugin package is not installed in the server's environment.
- `Redis at … is unreachable`: Redis down or wrong `MEMBRANE_REDIS_URL`.
- `Cannot open data directory`: volume not writable by UID 1000, or a
  bad `MEMBRANE_DATA_KEY_FILE`.

### Callers get 401 / 403

`401`: unknown or missing key. `403`: valid key without the route's
scope. Check the caller's key against the keyfile's scopes; see
[Security](../security.md).

### Callers get 429 or 503 with `Retry-After`

`429`: the caller exceeded `--rate-limit`. `503 {"error": "overloaded"}`:
all `--max-concurrency` slots were busy for 100 ms. Both are counted
in `membrane_requests_rejected_total{reason}`. Clients should honour
`Retry-After`. Persistent `overloaded` means the node needs more CPU
or more replicas, not a higher limit.

### Redis writes falling behind

`membrane_persistence_queue_depth` rising means Redis is slow or down;
the background writer retries with backoff while the node keeps
serving from memory. `membrane_persistence_dropped_total` counts writes
dropped because the queue (10,000 operations) was full or the node
stopped before flushing. Those fragments will be missing after a
restart, and their replicas on peers still serve them.

### A peer's circuit breaker is open

`membrane_peer_circuit_open{peer="..."}` is 1: five consecutive calls to
that peer failed, so calls fail fast for 30 s and then one trial call
is let through. Quorum writes stop waiting on that peer, and
replication and gossip skip it. Check the peer (`/livez`, logs); the
breaker closes on the first successful trial.

### Certificate expiring

`membrane_tls_cert_expiry_seconds` below 14 days: with
`--tls-cert/--tls-key`, replace the files (the node reloads them within
a minute, or `kill -HUP` it). ACME nodes renew 30 days ahead; a
shrinking value means renewal is failing: check the logs for `ACME`
and that port 80 of the domain reaches `--tls-acme-http-port`.

### Audit chain broken

`membrane_audit_chain_valid` is 0 and the log shows `audit log chain
is broken at entry N`: `<data-dir>/audit.jsonl` was edited, truncated,
or reordered. Preserve the file and the node's disk for investigation;
the node keeps recording new entries after the break.

### Memory pressure

`membrane_evictions_total{reason="capacity"}` rising while hit rates
fall means the working set does not fit. Raise `--max-memory`, add
nodes, or shorten fragment TTLs.

### Cluster partition

`/peers` on different nodes disagree, and
`membrane_gossip_failures_total` climbs. Strong writes fail closed on
the minority side, so no acknowledged write is lost. Fix the network;
nodes re-converge through heartbeats and gossip without intervention.

## Communication templates

```text
[SEV-X] <one-line summary>
Started: <UTC timestamp>
Impact: <user-facing impact>
Lead: <on-call name>
Updates: every 15 min in this thread.
```

```text
[SEV-X RESOLVED] <one-line summary>
Duration: <start> -> <end>
Root cause: <one paragraph>
Follow-ups: <links>
```

## After the incident

Within 5 business days write a review covering the timeline, root
cause, contributing factors, and 3–5 action items with owners.
