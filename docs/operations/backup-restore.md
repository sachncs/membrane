# Backup and Restore

Membrane's canonical state lives in Redis. This document covers backup,
restore, integrity check, and disaster recovery.

## Backup

Redis snapshots are the canonical backup mechanism. Membrane uses Redis's
RDB persistence (`SAVE` / `BGSAVE`) for point-in-time snapshots.

### Snapshot cron

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

### Retention

Keep hourly snapshots for 24 h, daily snapshots for 30 d, weekly snapshots
for 1 y. A small script can prune older files via S3 lifecycle policies.

## Restore

```bash
#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="${1:?usage: $0 <rdb-path>}"
REDIS_HOST="${MEMBRANE_REDIS_HOST:-redis}"
REDIS_PORT="${MEMBRANE_REDIS_PORT:-6379}"

# Stop Membrane so nothing writes during restore.
kubectl scale deployment membrane --replicas=0 -n membrane

# Copy snapshot into Redis.
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" SHUTDOWN NOSAVE || true
sleep 2
cp "$SNAPSHOT" /var/lib/redis/dump.rdb
chown redis:redis /var/lib/redis/dump.rdb
redis-server /etc/redis/redis.conf &

# Wait for Redis to come back.
until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping; do sleep 1; done

# Bring Membrane back.
kubectl scale deployment membrane --replicas=3 -n membrane
```

## Integrity check

After restore, run:

```python
from membrane.persistence.redis import Redis

r = Redis("redis://redis:6379/0")
digest = r.inventory_digest()
assert len(digest) > 0, "no fragments restored"
print(f"Restored {len(digest)} fragments")
```

## Disaster recovery

| Scenario | Recovery |
|---|---|
| Redis data lost, last snapshot > 24 h old | Restore from snapshot; expect data loss for the gap |
| Redis corrupted on disk | Replace pod with `redis-data` PVC intact; otherwise restore |
| Full cluster lost | Provision fresh Redis + Membrane; restore snapshot; re-attach nodes |

## References

* Redis persistence: https://redis.io/docs/management/persistence/
* S3 lifecycle policies: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html
