# Upgrade Procedure

This document covers Membrane version upgrades with a focus on zero-downtime
rollouts and clean rollback paths.

## Versioning

Membrane follows [Semantic Versioning](https://semver.org/):

* **Major** — breaking wire-format or API changes; requires migration window.
* **Minor** — backward-compatible feature additions; safe rolling upgrade.
* **Patch** — backward-compatible bug fixes; safe rolling upgrade.

The compatibility window is the current and previous minor (`N` and `N-1`).

## Rolling upgrade (minor/patch)

For a 3-replica deployment:

```bash
# 1. Update the image tag.
kubectl set image deployment/membrane \
    membrane=membrane:v0.3.0 \
    -n membrane

# 2. Watch the rollout. Each pod terminates only after the new pod is
#    healthy and has joined the cluster.
kubectl rollout status deployment/membrane -n membrane
```

Membrane's cluster layer uses an AP merge policy (`Fragment.merge` with
`max(version_id)`), so concurrent writes from old + new nodes converge
without loss. Gossip propagates the new view within one gossip interval.

## Major upgrade

Major upgrades include wire-format or schema-version changes. Steps:

1. Read `CHANGELOG.md` for the breaking changes.
2. Drain traffic from the cluster (`kubectl cordon` then `kubectl drain`).
3. Upgrade the Redis schema if applicable (see `membrane.serialization.SCHEMA_VERSION`).
4. Bring up the new pods.
5. Verify fragment hashes against the schema version (`from_dict` will
   raise `SchemaError` if mismatched).
6. Uncordon.

## Rollback

For a failed rollout, Kubernetes keeps the previous ReplicaSet:

```bash
kubectl rollout undo deployment/membrane -n membrane
kubectl rollout status deployment/membrane -n membrane
```

For a major-version rollback, also restore Redis from the pre-upgrade
snapshot (see `backup-restore.md`).

## Pre-upgrade checklist

- [ ] Backup taken (see `backup-restore.md`)
- [ ] CHANGELOG reviewed
- [ ] Schema version verified
- [ ] Compatibility window checked (current + previous minor supported)
- [ ] Smoke test plan documented
- [ ] Rollback plan documented and reviewed
