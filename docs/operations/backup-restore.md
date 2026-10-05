# Backup and Restore

Membrane is a cache: every fragment can be recomputed by running
prefill again. Backups therefore protect warm-cache performance, not
correctness. Decide first whether a cold start after a disaster is
acceptable; if it is, you only need replicas.

## What holds state

| State | Where | Needed to restore a node |
|-------|-------|--------------------------|
| Fragment metadata | Redis (`--redis`), keys prefixed `membrane:` | yes |
| KV bytes | `<data-dir>/blobs` on the node's volume (`--data-dir`), AES-256-GCM encrypted | yes |
| Data key | `<data-dir>/master.key`, or the file given with `--data-key-file` | yes; without it the blobs are unreadable |
| API keys, TLS material | your secret store | yes |

Without `--redis` and `--data-dir` a node keeps everything in memory
and restarts empty; peers still hold their replicas.

On startup a node with `--redis` reloads the fragments listed for its
node id, keeping those that are metadata-only or whose bytes are still
in its data directory. It drops the rest from its index.

## Backup

**Redis.** Use Redis persistence (AOF or RDB) and copy the dump files
off the host on a schedule, for example:

```bash
redis-cli -h "$REDIS_HOST" BGSAVE
# wait for LASTSAVE to change, then copy dump.rdb to object storage
```

**Data directory.** Snapshot each node's volume. In Kubernetes these
are the `data-membrane-N` PersistentVolumeClaims created by the
StatefulSet; use `VolumeSnapshot` objects or your storage provider's
snapshots. Under Docker Compose it is the `membrane-data` volume.

**Data key.** If the key was generated into the data directory, the
volume snapshot contains it, so protect snapshots like the key itself.
Production deployments should instead keep the key in a secret manager
and mount it with `--data-key-file` / `MEMBRANE_DATA_KEY_FILE`.

## Restore

1. Stop the nodes (`kubectl scale statefulset membrane --replicas=0`).
2. Restore Redis from its dump and wait for `redis-cli ping`.
3. Restore each node's data volume from its snapshot, matching the
   node id (`membrane-0` gets `data-membrane-0`).
4. Start the nodes. Each logs `Restored N fragments for <node> from Redis`.
5. Check with an admin key:

   ```bash
   membrane client inventory --base-url https://membrane-0... --api-key "$ADMIN_KEY"
   ```

## Disaster scenarios

| Scenario | Effect | Recovery |
|----------|--------|----------|
| One node lost | Its fragments are still on replicas | Replace the pod; it rejoins and fills up again |
| Redis lost, volumes intact | Nodes start empty (no metadata to reload) | Restore Redis, or accept a cold cache |
| Volume lost, Redis intact | Fragments without bytes are skipped at startup | Nothing; they are recomputed on demand |
| Data key lost | Blobs unreadable; `/retrieve` reports them `corrupt` | Delete the data directory and start cold |
| Whole cluster lost | Cold cache | Redeploy; optionally restore Redis and volumes first |
