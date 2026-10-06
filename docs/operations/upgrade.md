# Upgrades

Upgrade a Membrane deployment without downtime. Patch and minor releases roll out one node at a time while the cluster keeps serving; major releases are planned migrations. Read the [CHANGELOG](../../CHANGELOG.md) before every upgrade, because it lists breaking changes under a **Breaking** heading.

## Release types

Membrane follows [Semantic Versioning](https://semver.org/).

| Release | What changes | Procedure |
|---------|--------------|-----------|
| Patch | Bug fixes | [Rolling upgrade](#rolling-upgrade) |
| Minor | Features; operator-visible changes are called out | Rolling upgrade, after reading **Breaking** |
| Major | The wire format (`SCHEMA_VERSION`) or the API | [Planned migration](#major-upgrades) |

Nodes of adjacent minor versions interoperate during a rolling upgrade.

## Before you upgrade

- [ ] The CHANGELOG's **Breaking** section is read.
- [ ] A backup is taken, for major upgrades. See [Backup and restore](backup-restore.md).
- [ ] `quorum_count` leaves headroom for one node being replaced.
- [ ] The rollback command is ready.

## Rolling upgrade

### Kubernetes

```bash
kubectl set image statefulset/membrane membrane=ghcr.io/sachncs/membrane:X.Y.Z -n membrane
kubectl rollout status statefulset/membrane -n membrane
```

The StatefulSet replaces one pod at a time, highest ordinal first. Each pod drains on `SIGTERM` (readiness turns `503` and its primaries are handed off), keeps its data volume, restores its fragments from Redis on start, and rejoins its peers. The PodDisruptionBudget keeps two pods available.

With three nodes and the default `quorum_count` of 2, strong writes keep succeeding while one node is being replaced. CI verifies this on every change: a rolling restart under continuous strong writes completes with no failed writes.

### Docker Compose and systemd

```bash
git fetch --tags && git checkout vX.Y.Z
docker compose up -d --build membrane                                          # Docker Compose
uv sync --frozen --no-dev --extra server && sudo systemctl restart membrane    # systemd
```

## Release notes for operators

### Upgrading to the current release

- **Tenant-scoped keys.** Each tenant now keeps its own copy of a content hash. `/inventory`, persistence records, warm-tier entries, and fragment events name a non-`public` tenant's fragment `<tenant>:<hash>`. Records persisted by an earlier version are re-keyed when a node restores them.
- **`/admin/policy`** now reads and changes the live promotion thresholds, and answers `409` when promotion is off.
- **Vault:** `VaultSecretProvider` reads the `secret` mount by default (a `secret/data` prefix still works).

### Upgrading to the Python 3.14 release

This release requires Python 3.14. Rebuild virtual environments (`uv sync --frozen` installs 3.14 automatically) and images (the Dockerfile already uses `python:3.14-slim`). Before rolling it out:

- `chmod 600` your API keyfile, TLS private key, and data key. The server refuses secret files that other users can read.
- Optionally convert the keyfile to hashed lines. Plaintext lines still work but log a warning. For each key, `printf '%s' "$KEY" | sha256sum | cut -d' ' -f1` (on macOS, `shasum -a 256`) gives the digest for `sha256:<digest>:<subject>:<scopes>`.
- FastAPI's `/docs`, `/redoc`, and `/openapi.json` are served only with `--enable-api-docs`.
- The LMCache integration and the `vllm`, `sglang`, and `trtllm` extras were removed; see [Compatibility](../compat-matrix.md).
- Scripts that parse CLI output: results still go to stdout, while status and errors go to stderr as log records.

## Major upgrades

1. Take a backup ([Backup and restore](backup-restore.md)).
2. Read the CHANGELOG's migration notes.
3. Stop all nodes. Nodes of different major versions do not interoperate.
4. Convert stored envelopes if the notes say so. For v2 to v5: `python tools/upgrade_v2_to_v5.py`.
5. Deploy the new version, and verify with `membrane client inventory`.

## Roll back

```bash
kubectl rollout undo statefulset/membrane -n membrane
```

Within a major version, data written by the newer version is readable by the older one. Across major versions, restore the pre-upgrade backup.
