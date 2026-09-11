# Upgrade Procedure

This document covers Membrane version upgrades with a focus
on zero-downtime rollouts and clean rollback paths. The
canonical manifest is a **StatefulSet** (see
`deployment/k8s/statefulset.yaml`); rolling-upgrade and
rollback commands therefore target `statefulset/membrane`,
not `deployment/membrane`.

## Versioning

Membrane follows [Semantic Versioning](https://semver.org/):

* **Major** — breaking wire-format or API changes; requires
  migration window.
* **Minor** — backward-compatible feature additions; safe
  rolling upgrade.
* **Patch** — backward-compatible bug fixes; safe rolling
  upgrade.

The compatibility window is the current and previous minor
(`N` and `N-1`).

## Rolling upgrade (minor/patch)

For a 3-replica StatefulSet (`metadata.name: membrane`):

```bash
# 1. Update the image tag. The StatefulSet preserves the
#    PVCs bound to each pod so the encrypted blob store
#    survives the rolling restart.
kubectl set image statefulset/membrane \
    membrane=membrane:vX.Y.Z \
    -n membrane

# 2. Watch the rollout. Each pod terminates only after the
#    new pod is healthy and has joined the cluster.
kubectl rollout status statefulset/membrane -n membrane
```

Membrane's cluster layer uses an AP merge policy
(`Fragment.merge` with `max(version_id)`), so concurrent
writes from old + new nodes converge without loss. Gossip
propagates the new view within one gossip interval.

## Major upgrade

Major upgrades include wire-format or schema-version changes
(e.g. v2 -> v5). Steps:

1. Read `CHANGELOG.md` for the breaking changes.
2. Run `tools/upgrade_v2_to_v5.py` (and the JSON helper
   `tools/upgrade_v2_to_v5_json.py`) against the cluster's
   storage backend to convert any pre-v5 envelopes.
3. Drain traffic from the cluster (`kubectl cordon` then
   `kubectl drain`).
4. Upgrade the Redis schema if applicable (see
   `membrane.serialization.SCHEMA_VERSION`).
5. Bring up the new pods (`kubectl apply -f
   deployment/k8s/statefulset.yaml`).
6. Verify fragment hashes against the schema version
   (`from_dict` will raise `SchemaError` if mismatched).
7. Uncordon.

## Rollback

For a failed rollout, Kubernetes keeps the previous
StatefulSet revision:

```bash
kubectl rollout undo statefulset/membrane -n membrane
kubectl rollout status statefulset/membrane -n membrane
```

For a major-version rollback, also restore Redis from the
pre-upgrade snapshot (see `backup-restore.md`). The
StatefulSet's stable pod identities (`membrane-0`,
`membrane-1`, …) mean a rollback preserves the PVC bindings
for the encrypted blob store; only the running image changes.

## Persistent volume caveat

A StatefulSet upgrade preserves the PVCs, so the
encrypted-blob store at `/var/lib/membrane/blobs` (mounted
from each pod's `membrane-data-membrane-N` PVC) survives the
rolling restart. Operators rolling a **Deployment** instead
of the StatefulSet must export the blob root separately or
the new pods will come up empty.

## Pre-upgrade checklist

- [ ] Backup taken (see `backup-restore.md`)
- [ ] CHANGELOG reviewed
- [ ] Schema version verified
- [ ] Compatibility window checked (current + previous minor
      supported)
- [ ] Smoke test plan documented
- [ ] Rollback plan documented and reviewed
