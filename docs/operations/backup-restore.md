# Backup and Restore

Membrane v3.0.0 stores fragment payloads in a **dual-store**
architecture: Redis carries metadata (frame magic, identity
sub-dict, lifecycle counters) and the canonical payload
bytes live in the **encrypted FilesystemBlob** store at the
local filesystem root configured by ``FilesystemBlob(root=...)``
(``membrane/content_store.py:FilesystemBlob``). A disaster
recovery procedure that backs up only the Redis RDB will
silently lose every fragment because the bytes are not in
the Redis snapshot.

This document covers backup, restore, integrity check, and
disaster recovery for both halves of the store.

## Backup

Both halves of the dual-store must be backed up:

### Redis metadata snapshot

Redis snapshots are the canonical metadata backup
mechanism. Membrane uses Redis's RDB persistence
(`SAVE` / `BGSAVE`) for point-in-time snapshots.

```cron
# /etc/cron.d/membrane-backup
0 */6 * * * membrane /usr/local/bin/membrane-backup.sh
```

`membrane-backup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REDIS_HOST="${MEMBRANE_REDIS_HOST:-redis}"
REDIS_PORT="${MEMBRANE_REDIS_PORT:-6379}"
BUCKET="${MEMBRANE_BACKUP_BUCKET:-s3://membrane-backups/redis}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE
sleep 5
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --rdb /tmp/membrane-${TS}.rdb
aws s3 cp /tmp/membrane-${TS}.rdb "${BUCKET}/${TS}.rdb"
rm /tmp/membrane-${TS}.rdb
```

### Filesystem blob snapshot

The encrypted blob store lives under the directory passed
to ``FilesystemBlob(root=...)``; the v3.0.0 production default
is ``/var/lib/membrane/blobs`` on each node. Operators that
run Membrane as a Kubernetes StatefulSet should snapshot the
PVC named ``membrane-data-membrane-N`` for each pod; operators
that run it under docker-compose should snapshot the
``membrane-blobs`` bind mount.

```bash
#!/usr/bin/env bash
set -euo pipefail

BLOB_ROOT="${MEMBRANE_BLOB_ROOT:-/var/lib/membrane/blobs}"
BUCKET="${MEMBRANE_BACKUP_BUCKET:-s3://membrane-backups/blobs}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# Per-node incremental snapshot. Each node's blob root is
# independent because content-addressing makes the bytes
# dedup-able across the cluster; the snapshot tool should
# gzip the tarball to keep S3 costs low.
tar -C "$BLOB_ROOT" -czf "/tmp/membrane-blobs-${TS}.tar.gz" .
aws s3 cp "/tmp/membrane-blobs-${TS}.tar.gz" \
    "${BUCKET}/${TS}/blobs.tar.gz"
rm "/tmp/membrane-blobs-${TS}.tar.gz"
```

### Retention

Keep hourly snapshots for 24 h, daily snapshots for 30 d,
weekly snapshots for 1 y. A small script can prune older
files via S3 lifecycle policies.

## Restore

```bash
#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="${1:?usage: $0 <rdb-path> <blobs-tarball>}"
BLOB_TARBALL="${2:?usage: $0 <rdb-path> <blobs-tarball>}"
REDIS_HOST="${MEMBRANE_REDIS_HOST:-redis}"
REDIS_PORT="${MEMBRANE_REDIS_PORT:-6379}"
BLOB_ROOT="${MEMBRANE_BLOB_ROOT:-/var/lib/membrane/blobs}"

# Stop Membrane so nothing writes during restore.
kubectl scale statefulset membrane --replicas=0 -n membrane

# Restore Redis metadata.
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" SHUTDOWN NOSAVE || true
sleep 2
cp "$SNAPSHOT" /var/lib/redis/dump.rdb
chown redis:redis /var/lib/redis/dump.rdb
redis-server /etc/redis/redis.conf &

# Restore the encrypted blob root.
rm -rf "$BLOB_ROOT"
mkdir -p "$BLOB_ROOT"
tar -C "$BLOB_ROOT" -xzf "$BLOB_TARBALL"

# Wait for Redis to come back.
until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping; do sleep 1; done

# Bring Membrane back.
kubectl scale statefulset membrane --replicas=3 -n membrane
```

## Integrity check

After restore, run:

```python
from membrane.persistence.redis import Redis
from membrane.content_store import FilesystemBlob

r = Redis("redis://redis:6379/0")
blobs = FilesystemBlob("/var/lib/membrane/blobs")
digest = r.inventory_digest()
assert len(digest) > 0, "no fragments restored"
missing = [h for h in digest if blobs.get(h) is None]
assert not missing, f"{len(missing)} metadata entries have no blob"
print(f"Restored {len(digest)} fragments ({len(missing)} missing blobs)")
```

## Disaster recovery

| Scenario | Recovery |
|----------|----------|
| Redis data lost, blobs intact | Restore Redis RDB only; cluster picks up where it left off. |
| Redis intact, blob root lost | Re-create the blob root; cluster will surface `CorruptPayloadError` for any metadata entry whose payload_ref is unreadable. Re-ingest those fragments from source. |
| Redis data lost, last snapshot > 24 h old | Restore both Redis and blobs from snapshot; expect data loss for the gap. |
| Redis corrupted on disk | Replace pod with `redis-data` PVC intact; otherwise restore Redis from snapshot. |
| Blob root corrupted on disk | Restore the blob tarball; ``corrupt_payloads_total`` will spike during the restore as the integrity check walks every payload. |
| Full cluster lost | Provision fresh Redis + Membrane StatefulSet; restore Redis snapshot **and** blob tarball before re-attaching nodes. |

## References

* Redis persistence: https://redis.io/docs/management/persistence/
* Filesystem blob store: `membrane/content_store.py:FilesystemBlob`
* S3 lifecycle policies: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html
* `docs/operations/upgrade.md` — version-bump procedure
