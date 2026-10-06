# Deployment Guide

## Overview

This guide covers running Membrane as a service: a single node, a
container, Docker Compose, Kubernetes, and systemd. New to Membrane?
Start with the [Quickstart](getting-started.md).

## 1. Install from source

Membrane runs on Python 3.14 only.

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
uv sync --frozen --no-dev --extra server   # locked versions, Python 3.14
.venv/bin/membrane --version
```

Or with pip on Python 3.14, without cloning:
`pip install "membrane[server] @ git+https://github.com/sachncs/membrane.git@vX.Y.Z"`.

## 2. Configuration

Every `membrane serve` flag has a `MEMBRANE_*` environment variable
(`membrane serve --help` lists them; [`.env.example`](../.env.example)
is a starting point). The container image, Compose file, Kubernetes
manifests, and systemd unit are all configured this way.

## 3. Security model in one paragraph

`membrane serve` binds `127.0.0.1` by default. To listen on any other
address it requires inbound authentication: an API keyfile
(`--api-key-file`) or mTLS (`--tls-cert/--tls-key/--tls-ca`). Every
route except the `/livez` and `/readyz` probes then needs credentials
(`read` for reads and `/metrics`, `write` for writes, `admin` for
deletes and `/admin/*`). See [security.md](security.md).

Keyfile lines hold the SHA-256 digest of each key, never the key; the
subject is the tenant the key reads and writes. `membrane keys
generate --subject ingest-svc --scope read --scope write` prints a new
key and its line:

```text
sha256:3f9c...e1:ingest-svc:read,write
sha256:8a1b...07:metrics-scraper:read
sha256:c42d...9a:membrane-peers:admin
```

The keyfile must not be readable by other users (`chmod 600`); the
server refuses to start otherwise.

## 4. Capacity, shutdown, and logs

| Flag | Env | Default | Effect |
|------|-----|---------|--------|
| `--http-threads` | `MEMBRANE_HTTP_THREADS` | 1 (4 in the free-threaded image) | Event loops serving HTTP, one thread each, sharing the node; scales across cores on free-threaded Python (below) |
| `--max-concurrency` | `MEMBRANE_MAX_CONCURRENCY` | 64 | Requests handled at once, shared by all event loops; excess get `503` + `Retry-After` |
| `--rate-limit` / `--rate-limit-burst` | `MEMBRANE_RATE_LIMIT` / `..._BURST` | off | Requests per second per credential; excess get `429` |
| `--max-connections` | `MEMBRANE_MAX_CONNECTIONS` | unlimited | Open connections accepted |
| `--drain-timeout` | `MEMBRANE_DRAIN_TIMEOUT` | 30 | Seconds a `SIGTERM` drain may take |
| `--log-format` | `MEMBRANE_LOG_FORMAT` | `text` (`json` in the image) | One JSON object per log line, with `request_id` |
| `--transfer-compression` | `MEMBRANE_TRANSFER_COMPRESSION` | `zstd` | How KV bytes travel between nodes (`zstd`, `lz4`, `deflate`, `raw`); blobs of 8 MiB or more upload in resumable, verified chunks |
| `--kv-quantization` | `MEMBRANE_KV_QUANTIZATION` | `none` | Store KV tensors as `int8`, `fp8_e4m3`, `fp8_e5m2`, or `nf4` (lossy; 2-4x smaller; needs `membrane[transfer]`) |
| `--warm-tier-bytes` | `MEMBRANE_WARM_TIER_BYTES` | 0 | Keep fragments evicted from memory in an encrypted on-disk tier (`<data-dir>/warm`) up to this size; reads promote them back |
| `--content-store` | `MEMBRANE_CONTENT_STORE` | `filesystem` | `filesystem` (encrypted, durable), `encrypted-memory`, `memory`, or a plugin |
| `--placement` | `MEMBRANE_PLACEMENT` | `ring` | Policy answering `POST /route`: `ring`, `latency`, `selector`, `economic`, `joint`, or a plugin ([Memory API](memory-api.md)) |
| `--route-threshold` | `MEMBRANE_ROUTE_THRESHOLD` | 0 | Uncached prompt tokens above which `/route` says to offload prefill to Membrane; adapts to load (0: off) |
| `--promote-replicas` | `MEMBRANE_PROMOTE_REPLICAS` | 0 | Copy frequently read fragments to more peers, up to this many copies (0: off) |
| `--dynamic-roles` | `MEMBRANE_DYNAMIC_ROLES` | off | Re-evaluate and advertise the node's role from load |
| `--region` | `MEMBRANE_REGION` | none | Region advertised to peers (routing and replica locality) |
| `--origin` | `MEMBRANE_ORIGIN` | none | Run as a regional cache that reads misses through from this origin `HOST:PORT` |
| `--require-compat` | `MEMBRANE_REQUIRE_COMPAT` | none | Refuse fragments not stamped for `MODEL[:DTYPE]` |
| `--role` | `MEMBRANE_ROLE` | `both` | Disaggregation phases served under `/disagg`: `prefill`, `decode`, or `both` ([Prefill / decode](disaggregation.md)) |
| `--grpc-port` | `MEMBRANE_GRPC_PORT` | off | Also serve the prefill / decode RPCs on this port (needs `membrane[disagg]`; image: `--build-arg EXTRAS=disagg`) |

On `SIGTERM` (or Ctrl+C) a node drains: `/readyz` returns 503 so load
balancers stop routing to it, writes get 503 + `Retry-After`, primaries
are handed to peers, the node leaves the cluster, flushes queued Redis
writes, and exits 0. Give the process manager a stop timeout longer
than the drain timeout (the shipped Compose, Kubernetes, and systemd
files do).

## 5. Durability

By default a node keeps everything in memory and restarts empty (its
peers still hold replicas). To survive restarts, give each node both:

| Flag | Env | Stores |
|------|-----|--------|
| `--redis redis://host:6379/0` | `MEMBRANE_REDIS_URL` | Fragment metadata, written behind every store and removal by a background writer that retries through Redis outages |
| `--data-dir /var/lib/membrane` | `MEMBRANE_DATA_DIR` | KV bytes, AES-256-GCM encrypted, under `<data-dir>/blobs` |

The data key is generated into `<data-dir>/master.key` (mode 0600) on
first start; in production mount it from a secret manager with
`--data-key-file` / `MEMBRANE_DATA_KEY_FILE` (32 raw bytes or 64 hex
characters). A node refuses to start if `--redis` is set but Redis is
unreachable, rather than silently running without durability. See
[Backup & restore](operations/backup-restore.md).

## 6. Docker

```bash
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

The image is `python:3.14-slim` with the locked dependencies (no pip,
no curl). It runs as UID 1000 under `tini`, binds `0.0.0.0:8080`, logs
JSON, works with a read-only root filesystem (it needs a writable
`/tmp` only when mTLS is on), checks its own health with `python -m
membrane.healthcheck` (mTLS-aware), and drains on `SIGTERM`; stop it
with `docker stop -t 40`. All settings are `MEMBRANE_*` environment
variables (`membrane serve --help`).

### Using many cores: the free-threaded image

On a standard (GIL) Python, one node uses about one core for request
handling. `Dockerfile.ft` builds the same server on free-threaded Python
3.14 (`python3.14t`, PEP 703), where `--http-threads` event loops serve
requests in parallel against one shared node:

```bash
docker build -f Dockerfile.ft -t membrane:ft .
docker run -p 8080:8080 -e MEMBRANE_HTTP_THREADS=4 ... membrane:ft
```

Measured on a 12-core laptop:

| Workload | GIL build | Free-threaded |
|----------|-----------|---------------|
| `GET /retrieve` over HTTP (ApacheBench, keep-alive) | 6.1k req/s with 1 or 4 loops | 8.2k req/s with 1 loop, 15.4k with 2 (the load generator saturates beyond) |
| `Node.retrieve` from 4 threads vs 1 (in process) | 1.0x | 2.7x |

The read path takes no lock. Each request thread records reads in its own
buffer, and long-lived objects use deferred reference counting
(`membrane.runtime.concurrency.share`), so threads do not contend on a
shared reference count. `membrane_gil_enabled` (0 or 1) and
`membrane_http_event_loops` report what a node is running with. A
free-threaded node logs a warning if an extension turns the GIL back on.
grpcio is one such extension, so the free-threaded image serves `/disagg`
over REST only.

On a GIL build, scale up by running one node per core as cluster peers
rather than by adding event loops.

## 7. Docker Compose

[`docker-compose.yml`](../docker-compose.yml) runs one node behind an
nginx TLS edge with Redis persistence. Create the keyfile and an nginx
certificate first (commands are in the file header), then:

```bash
docker compose up --build -d   # Membrane + Redis + nginx, durable by default
curl -k -H "Authorization: Bearer <key>" https://localhost/inventory
```

## 8. Kubernetes

[`deployment/k8s/`](../deployment/k8s/) holds a 3-replica StatefulSet
with a headless Service for peer discovery, a PodDisruptionBudget, a
NetworkPolicy, and a ServiceMonitor. Each pod gets a 10 GiB
PersistentVolumeClaim (`data-membrane-N`) for its data directory.

1. Create the `membrane-secrets` Secret (`api-keys`, `peer-api-key`,
   `metrics-token`; see the template in `configmap.yaml`). `api-keys`
   holds the hashed lines from `membrane keys generate`; `peer-api-key`
   and `metrics-token` hold the keys themselves. The peer key needs the
   `admin` scope.
2. Set `MEMBRANE_PEER_NETWORKS` in `configmap.yaml` to your pod CIDR.
   Peer calls to private addresses are otherwise rejected by the SSRF
   guard.
3. `kubectl apply -f deployment/k8s/`.

Each pod advertises
`<pod>.membrane-headless.<namespace>.svc.cluster.local` to its peers.
On termination a pod pauses 5 s in `preStop` (so its endpoints are
removed), then drains within `MEMBRANE_DRAIN_TIMEOUT` (30 s) inside the
45 s `terminationGracePeriodSeconds`. A 3-replica rolling restart under
continuous strong writes completes with no failed writes.
With the defaults (`MEMBRANE_QUORUM_COUNT=2`) a strong write is
acknowledged once one peer holds a copy, and fails closed with `503`
when no peer is healthy.

## 9. Systemd Service (Linux)

```bash
sudo cp deployment/membrane.service /etc/systemd/system/
sudo install -d -m 0750 /etc/membrane
# MEMBRANE_* settings, one per line; at least MEMBRANE_HOST and
# MEMBRANE_API_KEY_FILE (or the MEMBRANE_TLS_* files).
sudoedit /etc/membrane/membrane.env
sudo systemctl daemon-reload
sudo systemctl enable --now membrane
sudo journalctl -u membrane -f
```

## 10. Multi-node checklist

- Every node needs a unique `MEMBRANE_NODE_ID` and an
  `MEMBRANE_ADVERTISE_HOST` its peers can resolve.
- API-key clusters: set `MEMBRANE_PEER_API_KEY_FILE` on every node; the
  key needs `admin` because peers replicate all tenants' fragments and
  propagate deletes.
- mTLS clusters: issue each node a certificate whose CN is its node id
  with a role prefix (e.g. `admin-membrane-0`) and whose extended key
  usage allows both server and client auth.
- Set `MEMBRANE_PEER_NETWORKS` to the peer CIDR.
- Size `MEMBRANE_QUORUM_COUNT` to at most the number of nodes.

## 11. Releases

Membrane is installed from source; it is not on PyPI. Tagged releases
(`.github/workflows/release.yml`) attach a wheel and sdist to the
GitHub Release and push `ghcr.io/sachncs/membrane:X.Y.Z`. See
[Release process](release.md).
