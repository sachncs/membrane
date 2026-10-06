# Backup and restore

Protect warm-cache performance through node, volume, and regional failures. Membrane is a cache: every fragment can be recomputed by running prefill again, so backups protect performance, not correctness. Decide first whether a cold start after a disaster is acceptable; if it is, replicas alone are enough.

## Recovery objectives

| Approach | Data loss on a node failure | Time to a warm cache |
|----------|-----------------------------|----------------------|
| Replicas only (default) | None while a replica survives | Immediate; the replacement node refills from peers and traffic |
| Replicas plus `--redis` and `--data-dir` | None for the node's own fragments | Minutes; the node reloads its fragments on start |
| Plus scheduled snapshots | Back to the last snapshot, even if the cluster is lost | Time to restore Redis and the volumes |

## What holds state

| State | Location | Needed to restore a node |
|-------|----------|--------------------------|
| Fragment metadata | Redis (`--redis`), keys prefixed `membrane:` | Yes |
| KV bytes | `<data-dir>/blobs` on the node's volume (`--data-dir`), AES-256-GCM encrypted | Yes |
| Data key | `<data-dir>/master.key`, or the file given with `--data-key-file` | Yes; without it the blobs are unreadable |
| API keys and TLS material | Your secret store | Yes |

Without `--redis` and `--data-dir`, a node keeps everything in memory and restarts empty, while its peers still hold their replicas.

On startup, a node with `--redis` reloads the fragments recorded for its node id. It keeps those that are metadata-only or whose bytes are still in its data directory, and drops the rest from its index.

## Back up

### Redis

Enable Redis persistence (AOF or RDB) and copy the dump off the host on a schedule:

```bash title="Snapshot Redis"
redis-cli -h "$REDIS_HOST" BGSAVE
# wait for LASTSAVE to change, then copy dump.rdb to object storage
```

### Data directory

Snapshot each node's volume. On Kubernetes these are the `data-membrane-N` PersistentVolumeClaims created by the StatefulSet; use `VolumeSnapshot` objects or your storage provider's snapshots. Under Docker Compose, it is the `membrane-data` volume.

### Data key

> [!WARNING]
> If the data key was generated into the data directory, every volume snapshot contains it, so a snapshot must be protected like the key itself. In production, keep the key in a secret manager and mount it with `--data-key-file` (`MEMBRANE_DATA_KEY_FILE`).

## Restore

1. Stop the nodes:

   ```bash
   kubectl scale statefulset membrane --replicas=0
   ```

2. Restore Redis from its dump, and wait for `redis-cli ping` to succeed.
3. Restore each node's data volume from its snapshot, matching the node id: `membrane-0` gets `data-membrane-0`.
4. Start the nodes. Each logs `Restored N fragments for <node> from Redis`.
5. Verify with an admin key:

   ```bash
   membrane client inventory --base-url https://membrane-0... --api-key "$ADMIN_KEY"
   ```

## Disaster scenarios

| Scenario | Effect | Recovery |
|----------|--------|----------|
| One node lost | Its fragments are still on replicas | Replace the pod; it rejoins and fills up again |
| Redis lost, volumes intact | Nodes start empty, with no metadata to reload | Restore Redis, or accept a cold cache |
| Volume lost, Redis intact | Fragments without bytes are skipped at startup | None needed; they are recomputed on demand |
| Data key lost | Blobs are unreadable; `/retrieve` reports them as corrupt | Delete the data directory and start cold |
| Whole cluster lost | Cold cache | Redeploy; optionally restore Redis and volumes first |
