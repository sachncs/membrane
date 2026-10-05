# Deployment Guide

## Overview

This guide covers running Membrane as a service: a single node, a
container, Docker Compose, Kubernetes, and systemd. New to Membrane?
Start with the [Quickstart](getting-started.md).

## 1. Install from source

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
pip install -e ".[server]"
membrane --version
```

Or without cloning:
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

Keyfile lines are `<key>:<subject>:<scope,...>`; the subject is the
tenant the key reads and writes:

```text
3f9c...e1:ingest-svc:read,write
8a1b...07:metrics-scraper:read
c42d...9a:membrane-peers:admin
```

## 4. Durability

By default a node keeps everything in memory and restarts empty (its
peers still hold replicas). To survive restarts, give each node both:

| Flag | Env | Stores |
|------|-----|--------|
| `--redis redis://host:6379/0` | `MEMBRANE_REDIS_URL` | Fragment metadata, written through on every store and removal |
| `--data-dir /var/lib/membrane` | `MEMBRANE_DATA_DIR` | KV bytes, AES-256-GCM encrypted, under `<data-dir>/blobs` |

The data key is generated into `<data-dir>/master.key` (mode 0600) on
first start; in production mount it from a secret manager with
`--data-key-file` / `MEMBRANE_DATA_KEY_FILE` (32 raw bytes or 64 hex
characters). A node refuses to start if `--redis` is set but Redis is
unreachable, rather than silently running without durability. See
[Backup & restore](operations/backup-restore.md).

## 5. Docker

```bash
docker build -t membrane:latest .
mkdir -p secrets && cp /path/to/api-keys secrets/
docker run --read-only --tmpfs /tmp -p 8080:8080 \
  -v "$PWD/secrets:/run/secrets:ro" \
  -e MEMBRANE_API_KEY_FILE=/run/secrets/api-keys \
  membrane:latest
curl -H "Authorization: Bearer <key>" localhost:8080/inventory
```

The image runs as UID 1000 under `tini`, binds `0.0.0.0:8080`, works with
a read-only root filesystem (it needs a writable `/tmp` only when mTLS is
on), and shuts down gracefully on `SIGTERM`. All settings are
`MEMBRANE_*` environment variables (`membrane serve --help`).

## 6. Docker Compose

[`docker-compose.yml`](../docker-compose.yml) runs one node behind an
nginx TLS edge with Redis persistence. Create the keyfile and an nginx
certificate first (commands are in the file header), then:

```bash
docker compose up --build -d   # Membrane + Redis + nginx, durable by default
curl -k -H "Authorization: Bearer <key>" https://localhost/inventory
```

## 7. Kubernetes

[`deployment/k8s/`](../deployment/k8s/) holds a 3-replica StatefulSet
with a headless Service for peer discovery, a PodDisruptionBudget, a
NetworkPolicy, and a ServiceMonitor. Each pod gets a 10 GiB
PersistentVolumeClaim (`data-membrane-N`) for its data directory.

1. Create the `membrane-secrets` Secret (`api-keys`, `peer-api-key`,
   `metrics-token`; see the template in `configmap.yaml`). The peer key
   must appear in `api-keys` with the `admin` scope.
2. Set `MEMBRANE_PEER_NETWORKS` in `configmap.yaml` to your pod CIDR.
   Peer calls to private addresses are otherwise rejected by the SSRF
   guard.
3. `kubectl apply -f deployment/k8s/`.

Each pod advertises
`<pod>.membrane-headless.<namespace>.svc.cluster.local` to its peers.
With the defaults (`MEMBRANE_QUORUM_COUNT=2`) a strong write is
acknowledged once one peer holds a copy, and fails closed with `503`
when no peer is healthy.

## 8. Systemd Service (Linux)

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

## 9. Multi-node checklist

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

## 10. Releases

Membrane is installed from source; it is not on PyPI. Tagged releases
(`.github/workflows/release.yml`) attach a wheel and sdist to the
GitHub Release and push `ghcr.io/sachncs/membrane:X.Y.Z`. See
[Release process](release.md).
