# Incident Response Playbook

This document covers the steps the on-call engineer takes when an alert
fires or an issue is reported. Pair with `slo.md` (alert thresholds) and
`backup-restore.md` (recovery actions).

## Severity levels

| Level | Definition | Response time |
|---|---|---|
| SEV-1 | Complete outage; data loss risk | < 15 min |
| SEV-2 | Major degradation; error budget burning | < 1 h |
| SEV-3 | Minor degradation; no error budget impact | < 4 h |

## On-call checklist

1. **Acknowledge** the alert within the response time for its severity.
2. **Page** the secondary if you cannot acknowledge within 5 min.
3. **Open** an incident ticket in the issue tracker with severity, summary,
   and start time.
4. **Communicate** in `#incidents`: post the ticket link + severity + first
   observations within 10 min.

## Triage

Run through these checks in order:

```bash
# 1. Process alive?
curl -fsS https://membrane.internal/livez

# 2. Capacity OK?
curl -fsS https://membrane.internal/readyz

# 3. Resource pressure?
curl -fsS https://membrane.internal/metrics | grep membrane_

# 4. Cluster health?
curl -fsS https://membrane.internal/peers
```

If `/livez` fails, the process is down. Check `kubectl get pods`,
`journalctl`, and the Docker logs.

If `/readyz` returns 503, the node is over capacity. Check
`membrane_memory_used_bytes` against `membrane_memory_limit_bytes`.

## Common scenarios

### Redis unreachable

1. Verify Redis: `kubectl exec redis-0 -- redis-cli ping`.
2. If Redis is down, restore from snapshot (see `backup-restore.md`).
3. Membrane falls back to the in-memory cache; check
   `membrane_persistence_operations_total{outcome="failure"}` for rate.

### Memory exhaustion

1. Check `membrane_evictions_total{reason=...}` — see which eviction
   reason dominates.
2. If capacity grew due to `expired`, tune TTLs.
3. If `lru` dominates, increase memory budget or shard the cluster.

### Cluster split-brain

1. Confirm with `/peers` — count healthy peers.
2. Look at `membrane_gossip_failures_total` for partition symptoms.
3. Use `kubectl cordon` to isolate a misbehaving node before fixing it.
4. The AP merge policy (`Fragment.merge` with `max(version_id)`) prevents
   stale reads after a heal; verify with `/metrics`.

## Comms templates

### Initial incident post

```
[SEV-X] <one-line summary>
Started: <UTC timestamp>
Impact: <user-facing impact>
Lead: <on-call name>
Ticket: <link>
Updates: every 15 min in this thread.
```

### Resolution post

```
[SEV-X RESOLVED] <one-line summary>
Duration: <start> -> <end> (Xh Ym)
Root cause: <one paragraph>
Mitigation: <what stopped the bleeding>
Followups: <links to action items>
```

## Post-incident review

Within 5 business days, file a post-incident review in `docs/postmortems/`
with: timeline, root cause, contributing factors, what went well, what to
improve, and 3-5 specific action items with owners and dates.
