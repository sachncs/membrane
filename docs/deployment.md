# Deployment

Run Membrane in production as a container, on Docker Compose, on Kubernetes, or under systemd. This guide covers installation, configuration, durability, and the settings every production node needs. New to Membrane? Start with the [Quickstart](getting-started.md).

## Choose a deployment

| Target | Best for | Starting point |
|--------|----------|----------------|
| [Container](#container) | A single node, or your own orchestration | `Dockerfile`, `Dockerfile.ft` |
| [Docker Compose](#docker-compose) | One host with TLS and persistence | [`docker-compose.yml`](../docker-compose.yml) |
| [Kubernetes](#kubernetes) | A replicated, self-healing cluster | [`deployment/k8s/`](../deployment/k8s/) |
| [systemd](#systemd) | Bare-metal or VM hosts | [`deployment/membrane.service`](../deployment/membrane.service) |

## Production checklist

Before a node takes traffic, confirm each item:

- [ ] **Authentication.** An API keyfile (`--api-key-file`) or mTLS. A node refuses to listen beyond loopback without one. See [Security](security.md).
- [ ] **TLS.** mTLS between nodes, or a TLS edge in front of clients.
- [ ] **Durability.** `--redis` for fragment metadata and `--data-dir` for KV bytes, so a restart does not start empty. See [Durability](#durability).
- [ ] **Data key.** Mounted from a secret manager with `--data-key-file`, not generated on the volume.
- [ ] **Peer settings.** A unique node id, a resolvable advertise host, a peer key with `admin` scope, and `MEMBRANE_PEER_NETWORKS`. See the [multi-node checklist](#multi-node-checklist).
- [ ] **Limits.** `--max-concurrency` and, for shared deployments, `--rate-limit`.
- [ ] **Shutdown.** A process-manager stop timeout longer than `--drain-timeout`.
- [ ] **Monitoring.** `/metrics` scraped and the alerts in [SLOs](operations/slo.md) configured.

## Install

Membrane runs on Python 3.14 only and is installed from source:

```bash title="Install a release from source"
git clone https://github.com/sachncs/membrane.git
cd membrane
uv sync --frozen --no-dev --extra server
.venv/bin/membrane --version
```

Or with pip on Python 3.14, without cloning:

```bash
pip install "membrane[server] @ git+https://github.com/sachncs/membrane.git@vX.Y.Z"
```

Tagged releases also publish a container image, `ghcr.io/sachncs/membrane:X.Y.Z`, and attach a wheel to the [GitHub release](https://github.com/sachncs/membrane/releases).

## Configure

Every `membrane serve` flag has a `MEMBRANE_*` environment variable. `membrane serve --help` lists them all, and [`.env.example`](../.env.example) is a starting point. The container image, Compose file, Kubernetes manifests, and systemd unit are all configured this way. The flags that matter most in production are listed in the [configuration reference](#configuration-reference) at the end of this page.

### Authentication in brief

`membrane serve` binds `127.0.0.1` by default. To listen on any other address, it requires an API keyfile (`--api-key-file`) or mTLS (`--tls-cert`, `--tls-key`, `--tls-ca`). Every route except the `/livez` and `/readyz` probes then needs credentials: `read` for reads and `/metrics`, `write` for writes, and `admin` for deletes and `/admin/*`.

A keyfile holds the SHA-256 digest of each key, never the key itself, and the key's subject is the tenant it reads and writes. `membrane keys generate --subject ingest-svc --scope read --scope write` prints a new key followed by its keyfile line:

```text title="api-keys"
sha256:3f9c...e1:ingest-svc:read,write
sha256:8a1b...07:metrics-scraper:read
sha256:c42d...9a:membrane-peers:admin
```

> [!IMPORTANT]
> The server refuses to start if the keyfile is readable by other users. Keep it `chmod 600`.

## Durability

By default a node keeps everything in memory and restarts empty; its peers still hold replicas. To survive restarts, give each node both of these:

| Flag | Environment variable | Stores |
|------|----------------------|--------|
| `--redis URL` | `MEMBRANE_REDIS_URL` | Fragment metadata, written behind every store and removal by a background writer that retries through Redis outages |
| `--data-dir PATH` | `MEMBRANE_DATA_DIR` | KV bytes, encrypted with AES-256-GCM, under `<data-dir>/blobs` |

`--redis` accepts three topologies, each with a `rediss+` form for TLS:

| URL | Topology |
|-----|----------|
| `redis://host:6379/0` | A single Redis server |
| `redis+sentinel://s1:26379,s2:26379/mymaster` | Redis Sentinel; follows the current master through failovers |
| `redis+cluster://n1:6379,n2:6379` | Redis Cluster |

The data key is generated into `<data-dir>/master.key` (mode 0600) on first start. In production, mount it from a secret manager with `--data-key-file` (`MEMBRANE_DATA_KEY_FILE`): 32 raw bytes or 64 hex characters.

> [!NOTE]
> If `--redis` is set and Redis is unreachable at startup, the node refuses to start rather than silently running without durability.

Backups and recovery are covered in [Backup and restore](operations/backup-restore.md).

## Graceful shutdown

On `SIGTERM` (or Ctrl+C) a node drains before it exits:

1. `/readyz` returns `503`, so load balancers stop routing to it.
2. Writes return `503` with `Retry-After`.
3. Its primaries are handed to peers, and it leaves the cluster.
4. Queued Redis writes are flushed, and it exits with status 0.

The drain is bounded by `--drain-timeout` (30 seconds). Give your process manager a stop timeout longer than that; the shipped Compose, Kubernetes, and systemd files already do.

## Container

```bash title="Build, create a key, and run"
docker build -t membrane:latest .
docker run --rm membrane:latest membrane keys generate --subject ops --scope admin > ops.txt
mkdir -p secrets && sed -n 2p ops.txt > secrets/api-keys && chmod 600 secrets/api-keys
sudo chown 1000 secrets/api-keys   # Linux: the image's user must own it
docker run --read-only --tmpfs /tmp -p 8080:8080 \
  -v "$PWD/secrets:/run/secrets:ro" \
  -e MEMBRANE_API_KEY_FILE=/run/secrets/api-keys \
  membrane:latest
curl -H "Authorization: Bearer $(sed -n 1p ops.txt)" localhost:8080/inventory
```

The image is hardened by default:

- `python:3.14-slim` with the locked dependencies, and no pip or curl.
- Runs as UID 1000 under `tini`, and works with a read-only root filesystem (it needs a writable `/tmp` only when mTLS is on).
- Binds `0.0.0.0:8080` and logs JSON.
- Checks its own health with `python -m membrane.healthcheck`, which is mTLS-aware.
- Drains on `SIGTERM`; stop it with `docker stop -t 40`.

### Free-threaded image

On a standard (GIL) Python build, one node uses about one core for request handling. `Dockerfile.ft` builds the same server on free-threaded Python 3.14 (`python3.14t`, PEP 703), where `--http-threads` event loops serve requests in parallel against one shared node:

```bash
docker build -f Dockerfile.ft -t membrane:ft .
docker run -p 8080:8080 -e MEMBRANE_HTTP_THREADS=4 ... membrane:ft
```

Measured on a 12-core laptop:

| Workload | GIL build | Free-threaded |
|----------|-----------|---------------|
| `GET /retrieve` over HTTP, keep-alive | 6.1k req/s with 1 or 4 loops | 8.2k req/s with 1 loop, 15.4k with 2 (the load generator saturates beyond) |
| `Node.retrieve`, 4 threads vs 1, in process | 1.0x | 3.2x (8.1M reads/s) |

The read path takes no lock: each thread reads through its own cache of the fragment table, and long-lived objects use deferred reference counting, so threads do not contend on shared state. The `membrane_gil_enabled` and `membrane_http_event_loops` metrics report how a node is running, and a free-threaded node logs a warning if an extension re-enables the GIL.

> [!NOTE]
> grpcio re-enables the GIL, so the free-threaded image serves `/disagg` over REST only. On a GIL build, scale up by running one node per core as cluster peers rather than by adding event loops.

## Docker Compose

[`docker-compose.yml`](../docker-compose.yml) runs one node behind an nginx TLS edge, with Redis persistence. Create the keyfile and an nginx certificate first (the commands are in the file's header), then:

```bash
docker compose up --build -d
curl -k -H "Authorization: Bearer <key>" https://localhost/inventory
```

## Kubernetes

[`deployment/k8s/`](../deployment/k8s/) contains a three-replica StatefulSet with a headless Service for peer discovery, a PodDisruptionBudget, a NetworkPolicy, and a ServiceMonitor. Each pod gets a 10 GiB PersistentVolumeClaim (`data-membrane-N`) for its data directory.

1. Create the `membrane-secrets` Secret with `api-keys`, `peer-api-key`, and `metrics-token` (see the template in `configmap.yaml`). `api-keys` holds the hashed lines from `membrane keys generate`; `peer-api-key` and `metrics-token` hold the keys themselves. The peer key needs the `admin` scope.
2. Set `MEMBRANE_PEER_NETWORKS` in `configmap.yaml` to your pod CIDR. Otherwise the outbound-request guard rejects peer calls to private addresses.
3. Apply the manifests:

   ```bash
   kubectl apply -f deployment/k8s/
   ```

Each pod advertises `<pod>.membrane-headless.<namespace>.svc.cluster.local` to its peers. On termination a pod pauses 5 seconds in `preStop` so its endpoints are removed, then drains within `MEMBRANE_DRAIN_TIMEOUT` (30 seconds), inside the 45-second `terminationGracePeriodSeconds`.

These behaviors are verified on every change by CI on a kind cluster:

- A rolling restart under continuous strong writes completes with **no failed writes**.
- After a pod is force-deleted, the remaining pods serve every byte it held.
- Scaling from 3 to 5 pods spreads primary ownership (no pod keeps more than 30%), and read capacity grows with the pod count. See [Capacity planning](operations/capacity.md).

With the defaults (`MEMBRANE_QUORUM_COUNT=2`), a strong write is acknowledged once one peer holds a copy, and fails closed with `503` when no peer is healthy.

## systemd

```bash title="Install the unit"
sudo cp deployment/membrane.service /etc/systemd/system/
sudo install -d -m 0750 /etc/membrane
sudoedit /etc/membrane/membrane.env
sudo systemctl daemon-reload
sudo systemctl enable --now membrane
sudo journalctl -u membrane -f
```

`/etc/membrane/membrane.env` holds `MEMBRANE_*` settings, one per line. Set at least `MEMBRANE_HOST` and `MEMBRANE_API_KEY_FILE` (or the `MEMBRANE_TLS_*` files).

## Multi-node checklist

- [ ] Every node has a unique `MEMBRANE_NODE_ID` and a `MEMBRANE_ADVERTISE_HOST` its peers can resolve.
- [ ] **API-key clusters:** `MEMBRANE_PEER_API_KEY_FILE` is set on every node. The key needs `admin`, because peers replicate every tenant's fragments and propagate deletes.
- [ ] **mTLS clusters:** each node's certificate has its node id as the CN with a role prefix (for example `admin-membrane-0`), and an extended key usage that allows both server and client authentication.
- [ ] `MEMBRANE_PEER_NETWORKS` covers the peer CIDR.
- [ ] `MEMBRANE_QUORUM_COUNT` is at most the number of nodes.

## Configuration reference

### Capacity and shutdown

| Flag | Environment variable | Default | Effect |
|------|----------------------|---------|--------|
| `--http-threads` | `MEMBRANE_HTTP_THREADS` | 1 (4 in the free-threaded image) | Event loops serving HTTP, sharing one node; scales across cores on free-threaded Python |
| `--max-concurrency` | `MEMBRANE_MAX_CONCURRENCY` | 64 | Requests handled at once across all loops; excess requests get `503` with `Retry-After` |
| `--rate-limit`, `--rate-limit-burst` | `MEMBRANE_RATE_LIMIT`, `MEMBRANE_RATE_LIMIT_BURST` | off | Requests per second per credential; excess requests get `429` |
| `--max-connections` | `MEMBRANE_MAX_CONNECTIONS` | unlimited | Open connections accepted |
| `--drain-timeout` | `MEMBRANE_DRAIN_TIMEOUT` | 30 | Seconds a `SIGTERM` drain may take |
| `--log-format` | `MEMBRANE_LOG_FORMAT` | `text` (`json` in the image) | `json` emits one object per line, with `request_id` |

### Data path

| Flag | Environment variable | Default | Effect |
|------|----------------------|---------|--------|
| `--content-store` | `MEMBRANE_CONTENT_STORE` | `filesystem` | Where KV bytes live: `filesystem` (encrypted, durable), `encrypted-memory`, `memory`, or a plugin |
| `--transfer-compression` | `MEMBRANE_TRANSFER_COMPRESSION` | `zstd` | How KV bytes travel between nodes: `zstd`, `lz4`, `deflate`, or `raw`. Blobs of 8 MiB or more upload in resumable, verified chunks |
| `--kv-quantization` | `MEMBRANE_KV_QUANTIZATION` | `none` | Store KV tensors as `int8`, `fp8_e4m3`, `fp8_e5m2`, or `nf4`: lossy, 2–4x smaller; needs `membrane[transfer]` |
| `--warm-tier-bytes` | `MEMBRANE_WARM_TIER_BYTES` | 0 | Keep fragments evicted from memory in an encrypted on-disk tier (`<data-dir>/warm`) up to this size; reads promote them back |
| `--require-compat` | `MEMBRANE_REQUIRE_COMPAT` | none | Refuse fragments not stamped for `MODEL[:DTYPE]` |

### Routing and topology

| Flag | Environment variable | Default | Effect |
|------|----------------------|---------|--------|
| `--placement` | `MEMBRANE_PLACEMENT` | `ring` | Policy behind `POST /route`: `ring`, `latency`, `selector`, `economic`, `joint`, or a plugin. See [Memory API](memory-api.md) |
| `--route-threshold` | `MEMBRANE_ROUTE_THRESHOLD` | 0 (off) | Uncached prompt tokens above which `/route` offloads prefill to Membrane; adapts to load |
| `--promote-replicas` | `MEMBRANE_PROMOTE_REPLICAS` | 0 (off) | Copy frequently read fragments to more peers, up to this many copies |
| `--dynamic-roles` | `MEMBRANE_DYNAMIC_ROLES` | off | Re-evaluate and advertise the node's role from its load |
| `--region` | `MEMBRANE_REGION` | none | Region advertised to peers, for routing and replica locality |
| `--origin` | `MEMBRANE_ORIGIN` | none | Run as a regional cache that reads misses through from this origin (`HOST:PORT`) |
| `--role` | `MEMBRANE_ROLE` | `both` | Disaggregation phases served under `/disagg`: `prefill`, `decode`, or `both`. See [Prefill and decode](disaggregation.md) |
| `--grpc-port` | `MEMBRANE_GRPC_PORT` | off | Also serve the prefill and decode RPCs on this port (needs `membrane[disagg]`; image: `--build-arg EXTRAS=disagg`) |

## Releases

Tagged releases, built by `.github/workflows/release.yml`, attach a wheel and sdist to the GitHub release and push `ghcr.io/sachncs/membrane:X.Y.Z`. See [Release process](release.md).
