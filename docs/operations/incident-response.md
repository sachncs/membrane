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

`/readyz` returns `503` only when the node object is missing; a full
node is ready. If `/livez` fails, the process is down: check
`kubectl get pods`, `kubectl logs`, or `journalctl -u membrane`.

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

The process exits with code 2 and prints the reason. Common ones:

- `Refusing to serve unauthenticated`: no API keyfile or TLS configured.
- `Cannot read API keyfile` / `contains no valid keys`: secret not
  mounted, or empty.
- `Redis at … is unreachable`: Redis down or wrong `MEMBRANE_REDIS_URL`.
- `Cannot open data directory`: volume not writable by UID 1000, or a
  bad `MEMBRANE_DATA_KEY_FILE`.

### Callers get 401 / 403

`401`: unknown or missing key. `403`: valid key without the route's
scope. Check the caller's key against the keyfile's scopes; see
[Security](../security.md).

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
