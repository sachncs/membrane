# Upgrade Procedure

Membrane follows [Semantic Versioning](https://semver.org/). Read the
[CHANGELOG](../../CHANGELOG.md) before every upgrade: it lists breaking
changes under a **Breaking** heading.

| Release | What changes | Procedure |
|---------|--------------|-----------|
| Patch | Bug fixes | Rolling upgrade |
| Minor | Features; operator-visible changes are called out | Rolling upgrade after reading **Breaking** |
| Major | Wire format (`SCHEMA_VERSION`) or API | Planned migration below |

Nodes of adjacent minor versions interoperate during a rolling upgrade.

## Rolling upgrade (Kubernetes)

```bash
kubectl set image statefulset/membrane membrane=ghcr.io/sachncs/membrane:X.Y.Z -n membrane
kubectl rollout status statefulset/membrane -n membrane
```

The StatefulSet replaces one pod at a time, highest ordinal first.
Each pod shuts down gracefully on `SIGTERM`, keeps its data volume
(`data-membrane-N`), restores its fragments from Redis on start, and
rejoins its peers. The PodDisruptionBudget keeps two pods available.

During the rollout one node is briefly missing. With the default
`quorum_count: 2` and three nodes, strong writes keep succeeding.

## Rolling upgrade (Docker Compose / systemd)

```bash
git fetch --tags && git checkout vX.Y.Z
docker compose up -d --build membrane          # Compose
pip install -e ".[server]" && sudo systemctl restart membrane   # systemd
```

## Major upgrade

1. Take a backup ([Backup & restore](backup-restore.md)).
2. Read the CHANGELOG's migration notes.
3. Stop all nodes. Mixed majors do not interoperate.
4. Convert stored envelopes if the notes say so (for v2 to v5:
   `python tools/upgrade_v2_to_v5.py`).
5. Deploy the new version and verify with `membrane client inventory`.

## Rollback

```bash
kubectl rollout undo statefulset/membrane -n membrane
```

Within a major version, data written by the newer version is readable
by the older one. Across majors, restore the pre-upgrade backup.

## Checklist

- [ ] CHANGELOG **Breaking** section read
- [ ] Backup taken (major upgrades)
- [ ] `quorum_count` leaves headroom for one node being replaced
- [ ] Rollback command ready
