<p align="center">
  <h1 align="center">Membrane</h1>
  <p align="center">Global Contextual Memory Fabric for distributed, content-addressed KV-cache sharing across LLM serving clusters.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/membrane/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/membrane/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/membrane/"><img src="https://img.shields.io/pypi/v/membrane" alt="PyPI"></a>
    <a href="https://github.com/sachncs/membrane/stargazers"><img src="https://img.shields.io/github/stars/sachncs/membrane" alt="Stars"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-strict-green.svg" alt="Checked with mypy"></a>
  </p>
</p>

**Membrane** is a Python library and runtime for distributed,
content-addressed KV-cache sharing across LLM serving clusters. It
implements the analytical throughput model and routing policy from
the paper [“Prefill-as-a-Service: KVCache of Next-Generation Models
Could Go Cross-Datacenter”](https://arxiv.org/abs/2604.15039),
separates the KV cache from GPU memory, and distributes it across a
cluster with consistent hashing, gossip-based membership, and
reconstruction-driven retrieval.

---

## Features

- **Analytical throughput model** — Verbatim reproduction of Equations (1)–(6) from the paper, with a piecewise-linear Table-5 fit
- **Throughput-optimal configuration** — Grid search over routing threshold and PD-split ratio
- **Dual-timescale scheduler** — Bandwidth-aware short-term routing and long-term reallocation
- **Content-addressed fragments** — Immutable KV segments keyed by hash with structural signatures
- **Four in-memory indices** — Exact, semantic, positional, and co-access lookup over the same fragment set
- **Reconstruction engine** — Context rebuild from fragments with prefill fallback when coverage is incomplete
- **Multi-tenant isolation** — Per-tenant policies with shared deduplicated canonical store
- **Cluster membership and gossip** — Heartbeats, failure detection, gossip state exchange, background replication
- **Consistent hashing + sharding** — Primary/replica placement with rebalancing on topology changes
- **Pluggable compute backends** — CPU, GPU (PyTorch), Transformers, OpenAI, Anthropic, Ollama
- **Authenticated HTTP transport** — FastAPI over HTTP or mTLS, with per-route scopes (API keys or client certificates) and per-tenant read isolation
- **Redis persistence** — Optional durability with LRU eviction and inventory digests
- **CLI with TUI dashboard** — Live monitoring, cluster status, and interactive setup wizard

---

## Installation

### From PyPI

```bash
pip install membrane
```

### From source

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
pip install -e ".[dev]"
```

### Optional extras

```bash
# Server dependencies (FastAPI, Redis, httpx, cryptography for TLS / encryption at rest)
pip install -e ".[server]"

# KV transfer engine and quantization (numpy, lz4, zstandard)
pip install -e ".[transfer]"

# Secret backends and tracing
pip install -e ".[secrets-aws]"   # or secrets-gcp, secrets-vault
pip install -e ".[otel]"

# GPU compute backend (PyTorch with CUDA)
pip install -e ".[gpu]"

# Local LLM backend (HuggingFace Transformers + tokenizers)
pip install -e ".[local-llm]"
```

**Requirements**: Python 3.10+ (3.10, 3.11, 3.12, 3.13 supported on CI)

---

## Quick Start

### CLI

```bash
# Reproduce the paper's analytical evaluation.
python scripts/demo.py
python scripts/demo_full.py
python scripts/demo_membrane.py
python scripts/demo_quantization.py

# Start a single-node server with the CPU backend (binds 127.0.0.1).
membrane serve --node-id n1 --port 8080 --compute cpu

# Expose it on the network: authentication is required.
printf '%s:ops:admin\n' "$(openssl rand -hex 32)" > api-keys
membrane serve --host 0.0.0.0 --api-key-file api-keys

# Open a live TUI dashboard against a running server.
membrane dashboard --host localhost --port 8080

# Show cluster membership.
membrane cluster-status

# Show LLM-backend status.
membrane llm-status

# Inspect / export the static configuration and OpenAPI spec.
membrane config
membrane config --openapi > docs/openapi.json

# Admin operations against a running node.
membrane admin snapshot
membrane admin rotate-keys

# One-off interactions with a running Membrane server.
membrane client store --content-hash abc...
membrane client retrieve --content-hash abc...
```

The full subcommand surface:

| Command | Purpose |
|---------|---------|
| `membrane serve` | Start a Membrane production server. |
| `membrane dashboard` | Live TUI dashboard against a remote server. |
| `membrane cluster-status` | Show cluster membership and peer health. |
| `membrane llm-status` | Show LLM backend status and model info. |
| `membrane config` | Show static configuration; `--openapi` writes the v3 spec. |
| `membrane admin` | Admin operations: `snapshot`, `restore`, `rotate-keys`. |
| `membrane client` | One-off `store` / `retrieve` / `inventory` calls. |

### Python API

```python
import membrane
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.node import Node

# Verify install and inspect the public surface.
print(f"{len(membrane.__all__)} exports available")

# Build a fragment identity (the durable content-addressed fingerprint).
identity = PayloadIdentity(
    payload_hash="placeholder",
    model_id="llama-3",
    model_revision="",
    tokenizer_name="llama-3",
    tokenizer_revision="",
    layer_range=(0, 1),
    head_range=(-1, -1),
    token_span=(0, 128),
    dtype="float16",
    shape=(1, 1, 1, 128, 64),
)
frag = Fragment(
    identity=identity,
    payload_ref="blob-key",
    payload_size=8 * 1024 * 1024,
    ttl=3600.0,
    reuse_score=0.5,
    version_id=1,
)

node = Node("n1", max_memory_bytes=1 << 30)
node.store(frag, is_primary=True)
retrieved = node.fragments.get(frag.identity.payload_hash)
```

### Docker

The image is configured entirely through `MEMBRANE_*` environment
variables and refuses to start without authentication.

```bash
docker build -t membrane .
docker run --read-only --tmpfs /tmp -p 8080:8080 \
  -v "$PWD/secrets:/run/secrets:ro" \
  -e MEMBRANE_API_KEY_FILE=/run/secrets/api-keys membrane
```

[`docker-compose.yml`](docker-compose.yml) runs Membrane behind an nginx TLS
edge with Redis persistence; its header lists the two files to create first.
Kubernetes manifests for a 3-replica cluster live in
[`deployment/k8s/`](deployment/k8s/).

---

## Configuration

Every `membrane serve` flag can also be set through the environment
variable shown; `membrane serve --help` is the authoritative list.

### Server

| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--node-id` | `MEMBRANE_NODE_ID` | `membrane-0` | Unique node identifier |
| `--host` | `MEMBRANE_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for all interfaces) |
| `--port` | `MEMBRANE_PORT` | `8080` | Listen port |
| `--compute` | `MEMBRANE_COMPUTE` | `cpu` | `cpu`, `gpu`, `ollama`, `openai`, `anthropic`, `transformers` |
| `--redis` | `MEMBRANE_REDIS_URL` | _disabled_ | Redis URL, e.g. `redis://localhost:6379/0` |
| `--max-memory` | `MEMBRANE_MAX_MEMORY` | `1<<30` | Per-node memory budget in bytes |
| `--daemon` | `MEMBRANE_DAEMON` | `false` | No TUI dashboard (implied when stdout is not a TTY) |
| `--log-level` | `MEMBRANE_LOG_LEVEL` | `INFO` | Logging level |

### Security

The server refuses to listen on a non-loopback address unless one of the
authentication modes below is configured (or `--allow-unauthenticated` is
passed). Every route except `/livez` and `/readyz` then requires
credentials: reads need the `read` scope, writes `write`, deletes and
`/admin/*` `admin`. Failures return `401` / `403`.

| Flag | Env Variable | Description |
|------|--------------|-------------|
| `--api-key-file` | `MEMBRANE_API_KEY_FILE` | Keyfile, one `<key>:<subject>:<scope,...>` per line. Clients send `Authorization: Bearer <key>`; `<subject>` is the tenant the key reads and writes. |
| `--tls-cert` / `--tls-key` / `--tls-ca` | `MEMBRANE_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | Serve HTTPS and require client certificates signed by the CA. The certificate must allow both server and client auth (it is also used for peer calls). |
| `--tls-allowed-cn` | `MEMBRANE_TLS_ALLOWED_CNS` | Client certificate CNs to accept. Scopes come from the CN prefix (`admin-`, `write-`, `read-`). |
| `--allow-unauthenticated` | `MEMBRANE_ALLOW_UNAUTHENTICATED` | Serve without auth on a public address (development only). |

### Cluster and replication

| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--peer` | `MEMBRANE_PEERS` | _none_ | Seed peer `host:port` (repeatable or comma-separated) |
| `--advertise-host` | `MEMBRANE_ADVERTISE_HOST` | bind host / FQDN | Address peers dial to reach this node |
| `--peer-network` | `MEMBRANE_PEER_NETWORKS` | _none_ | CIDR of the peer network, exempt from the SSRF private-address block |
| `--peer-api-key-file` | `MEMBRANE_PEER_API_KEY_FILE` | _none_ | Bearer key presented to peers (API-key clusters; needs `admin`) |
| `--consistency` | `MEMBRANE_CONSISTENCY` | `strong` | Default write consistency: `strong`, `quorum`, `eventual` |
| `--quorum-count` | `MEMBRANE_QUORUM_COUNT` | `2` | Copies (local included) a strong write waits for |
| `--replica-count` | `MEMBRANE_REPLICA_COUNT` | `2` | Replicas per primary shard |
| `--heartbeat-interval` | `MEMBRANE_HEARTBEAT_INTERVAL` | `2.0` | Heartbeat period in seconds |
| `--gossip-interval` | `MEMBRANE_GOSSIP_INTERVAL` | `5.0` | Gossip period in seconds |
| `--failure-remove-threshold` | `MEMBRANE_FAILURE_REMOVE_THRESHOLD` | `4` | Missed heartbeats before removing a peer |

A strong write fails closed with `503` when fewer than `quorum_count - 1`
healthy peers acknowledge it.

### LLM backends

| Flag | Env Variable | Description |
|------|--------------|-------------|
| `--llm-url` | `MEMBRANE_LLM_URL` | Base URL (used for Ollama or a custom OpenAI endpoint) |
| `--llm-model` | `MEMBRANE_LLM_MODEL` | Model identifier (e.g. `llama3.2`, `gpt-4o-mini`) |
| `--api-key` | `MEMBRANE_LLM_API_KEY` | API key for the OpenAI / Anthropic compute backend |

See [`.env.example`](.env.example) for a starting point.

---

## API

The exported names below map 1-to-1 to `membrane.__all__`. Each
entry links to the actual class; the README does not introduce
any non-exported names.

| Symbol | Type | Description |
|--------|------|-------------|
| `Fragment` | class | Content-addressed KV segment with a `PayloadIdentity` |
| `PayloadIdentity` | dataclass | Stable ten-field fragment fingerprint |
| `Prefix` / `Segment` / `Artifact` / `Trace` | classes | Memory objects layered on top of `Fragment` |
| `FragmentKind` | enum | Discriminator across memory-object kinds |
| `Node` / `Origin` / `Replica` | classes | Serving-plane roles |
| `Index` | class | Facade over the four sub-indexes (import sub-indexes from `membrane.exacts` etc.) |
| `Ring` / `Shard` | classes | Consistent-hashing placement |
| `Reconstructor` | class | Rebuilds a context from fragments; falls back to prefill |
| `TransferService` | class | Unified in-process + remote transfer plane |
| `PersistenceBackend` / `Memory` / `Redis` / `CachingPersistence` | classes / protocol | Pluggable storage backends |
| `Backend` / `CPU` / `GPU` / `Transformers` / `OpenAI` / `Anthropic` / `Ollama` | classes | Compute backends |
| `FastAPIServer` | class | FastAPI HTTP transport |
| `Server` | class | Unified runnable server (CLI + transports) |
| `Authenticator` | protocol | Authentication contract |
| `Error` + typed hierarchy | exceptions | `NetworkError`, `SchemaError`, etc. |
| `configure_logging` | function | Shared logging setup |

The decision / policy / analytical classes (`Economic`, `Latency`,
`Joint`, `Promotion`, `Offload`, `Isolation`, `Tenant`,
`Selector`, `Roles`, `Predict`, `Workload`) live under
`membrane.analytical` and are imported from there; they are
intentionally **not** re-exported at the package root.

---

## Examples

```bash
# 1. Reproduce the paper's analytical throughput curves.
python scripts/demo.py

# 2. Run the dual-timescale scheduler simulator end-to-end.
python scripts/demo_full.py

# 3. Spin up a single-node server with the TUI dashboard.
membrane serve --node-id n1 --port 8080 --compute cpu
membrane dashboard --host localhost --port 8080

# 4. Three-node local cluster with the CPU backend.
membrane serve --node-id n1 --port 8080 --daemon --peer localhost:8081 --peer localhost:8082
membrane serve --node-id n2 --port 8081 --daemon --peer localhost:8080 --peer localhost:8082
membrane serve --node-id n3 --port 8082 --daemon --peer localhost:8080 --peer localhost:8081
membrane cluster-status --host localhost --port 8080
```

The [`docs/`](docs/) directory hosts architecture notes, deployment recipes,
and an FAQ.

---

## Project Structure

```
membrane/
├── membrane/                          # SDK package
│   ├── __init__.py                    # Public API exports
│   ├── fragment.py                    # Core fragment data model
│   ├── prefix.py                      # Token-sequence memory object
│   ├── kv_segment.py                  # Per-layer KV slice
│   ├── artifact.py                    # Retrieved document/embedding
│   ├── tool_trace.py                  # Structured tool output
│   ├── memory_object.py               # MemoryObject protocol
│   ├── structural_signature.py        # Token-span + layer metadata
│   ├── fragmentation_engine.py        # Windowing, split, merge
│   ├── fragment_store.py              # Tiered-eviction content store
│   ├── fragment_graph.py              # Typed fragment relationship graph
│   ├── graph_manager.py               # Graph lifecycle + prefetch hints
│   ├── weighted_graph.py              # Weighted-edge graph + subclusters
│   ├── semantic_hash.py               # LSH-style similarity hash
│   ├── exact_index.py                 # content_hash -> entry index
│   ├── semantic_index.py              # Brute-force cosine similarity
│   ├── positional_index.py            # AVL-backed overlap / adjacency
│   ├── co_access_index.py             # Co-access graph
│   ├── index_system.py                # Aggregate facade over all four
│   ├── interval_tree.py               # Self-balancing AVL tree
│   ├── lru_cache.py                    # Access-tracker with eviction
│   ├── semantic_cluster.py            # Greedy similarity clustering
│   ├── subgraph_retrieval.py          # BFS over weighted graph
│   ├── shard_manager.py               # Consistent-hash shard assignment
│   ├── hash_ring.py                   # Karger-style consistent hashing
│   ├── supernode.py                    # Directory super-peer
│   ├── distributed_directory.py       # Multi-supernode directory
│   ├── global_directory.py            # Routing-plane registry
│   ├── membrane_node.py               # In-memory fragment storage
│   ├── origin_node.py                 # Canonical authority + replication
│   ├── replica_node.py                # Hot regional cache
│   ├── session_tracker.py             # Per-session access history
│   ├── workload_analyzer.py          # Pattern detection over logs
│   ├── node_selector.py               # Multi-criteria node selection
│   ├── node_telemetry.py              # Telemetry snapshot
│   ├── economic_router.py             # argmax(value_density - cost)
│   ├── latency_router.py              # Lowest-latency holder
│   ├── joint_optimizer.py             # Compute + memory placement
│   ├── offload_decision_engine.py     # Local vs remote prefill
│   ├── promotion_policy.py            # Multi-region replication
│   ├── predictor.py                   # Lightweight KV/reuse predictor
│   ├── reconstruction_engine.py       # Context rebuild from fragments
│   ├── prefill_adapter.py             # Model profiler integration
│   ├── remote_prefill_dispatcher.py   # Single-target dispatch
│   ├── async_prefill_dispatcher.py    # Concurrent race + fallback
│   ├── kv_transfer_after_prefill.py   # Ship KV back to requester
│   ├── kv_cache_manager.py            # Hit/miss tracked cache
│   ├── kv_segment.py                  # KV slice memory object
│   ├── cluster_replicator.py          # Replicate connected components
│   ├── delta_sync.py                  # Version-aware delta sync
│   ├── delta_encoder.py               # Prefix deltas (encode/decode)
│   ├── canonical_store.py             # Multi-tenant dedup store
│   ├── tenant_isolation.py            # Cross-tenant sharing policy
│   ├── chunked_transfer.py            # Chunked fragment transfer
│   ├── transfer_service.py            # Local-node transfer plane
│   ├── cache_metrics.py               # Immutable hit-rate counter
│   ├── cost_model.py                  # Recompute vs reuse cost
│   ├── value_density.py               # importance × expected reuse
│   ├── prefix_version_chain.py        # Append-only version chain
│   ├── dynamic_role_manager.py        # Role switching (memory/prefill/decode)
│   ├── logging.py                     # Shared logging configuration
│   ├── protocols.py                   # Structural Protocol interfaces
│   ├── server.py                      # Unified Membrane server
│   └── cli.py                         # Typer + Rich CLI / TUI dashboard
├── membrane/compute/                  # Compute backends
│   ├── __init__.py                    # Lazy optional-backend registry
│   ├── backend.py                     # ComputeBackend protocol
│   ├── cpu_backend.py                 # CPU reference implementation
│   ├── gpu_backend.py                 # PyTorch CUDA backend (with CPU fallback)
│   ├── transformers_backend.py        # HuggingFace Transformers backend
│   ├── openai_backend.py              # OpenAI REST backend
│   ├── anthropic_backend.py           # Anthropic REST backend
│   └── ollama_backend.py              # Ollama local server backend
├── membrane/persistence/              # Storage backends
│   ├── __init__.py                    # Public re-exports
│   ├── memory_backend.py              # In-memory backend (default)
│   └── redis_backend.py               # Redis-backed persistence
├── membrane/transport/                # Network transports
│   ├── __init__.py                    # Public re-exports
│   ├── fastapi.py                     # FastAPI + uvicorn server
│   ├── routes_fastapi.py              # Route bindings + per-route auth
│   ├── ops.py / ops_cluster.py        # Transport-agnostic operations
│   ├── authz.py                       # Route -> scope table
│   ├── tls.py / tls_protocol.py       # mTLS config + verified peer CN
│   └── admin.py                       # /admin/* router
├── membrane/network/                  # Peer-to-peer networking
│   ├── __init__.py                    # Public re-exports
│   ├── config.py                      # ClusterConfig dataclass
│   ├── cluster_manager.py             # Membership + gossip + replication
│   ├── gossip_state.py                # Gossip payload + merge
│   ├── peer_client.py                 # urllib-based peer client
│   └── remote_transfer.py             # Network-aware TransferService
├── membrane/model/                    # Analytical model + simulator
│   ├── __init__.py
│   ├── throughput_model.py            # Equations (1)–(6)
│   ├── profiler.py                    # KV size + prefill time estimators
│   ├── workload.py                    # Log-normal workload generator
│   ├── router.py                      # Length-based routing policy
│   ├── optimizer.py                   # Grid-search optimizer
│   ├── scheduler.py                   # Dual-timescale scheduler
│   ├── simulator.py                   # End-to-end simulation harness
│   └── metrics.py                     # TTFT and bandwidth metrics
├── tests/                             # Test suite (548+ tests)
├── scripts/                           # Demo and helper scripts
├── docs/                              # Architecture, deployment, FAQ
├── deployment/                        # systemd and nginx configs
├── docker-compose.yml                 # Multi-service local stack
├── Dockerfile                         # Container image
└── pyproject.toml                     # Build + tool configuration
```

---

## Development

```bash
# Install with dev dependencies.
pip install -e ".[dev]"

# Run the full test suite.
pytest tests/ -v

# Run only the model-layer tests.
pytest tests/test_optimizer.py tests/test_simulator.py tests/test_workload.py \
        tests/test_router.py tests/test_scheduler.py tests/test_throughput_model.py \
        tests/test_profiler.py

# Lint.
ruff check membrane/ tests/

# Format.
ruff format --check membrane/ tests/
ruff format membrane/ tests/

# Type check.
python -m mypy membrane/

# Run a paper-reproduction demo.
python scripts/demo.py
python scripts/demo_full.py

# Run with coverage.
pytest tests/ --cov=membrane --cov-report=term-missing

# Helper scripts.
bash scripts/setup.sh
bash scripts/cleanup.sh
```

### Code Style

- Line length: 120
- Quotes: double (`"`)
- Formatting: ruff (auto-format with `ruff format`)
- Type hints: required on all public signatures
- Docstrings: Google-style with "what" and "why"
- No semi-private naming (`_foo`) — all identifiers are public

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add weighted graph co-access predictor
fix: handle edge case in failure detection
docs: add comprehensive docstrings across all modules
refactor: convert semi-private attributes to public API
test: add parity tests for cache vs streamed memory
chore: update ruff config
```

---

## Testing

```bash
# Full unit + integration suite.
pytest tests/ -v

# With coverage.
pytest tests/ --cov=membrane --cov-report=term-missing

# Benchmark smoke (3.0.0 baseline perf).
pytest tests/bench/ -v

# Stress suite (64-thread concurrency; gated by the dedicated
# 'stress' CI job).
pytest tests/membrane/stress -v -m stress

# Chaos suite (toxiproxy-aware; the in-process _FakeProxy suite
# runs without an external proxy).
pytest tests/membrane/chaos -v -m chaos
```

---

## Documentation

The [`docs/`](docs/) directory is the project's documentation
hub. The full surface:

| Page | Purpose |
|------|---------|
| [`docs/getting-started.md`](docs/getting-started.md) | Five-minute onboarding walkthrough. |
| [`docs/architecture.md`](docs/architecture.md) | Module breakdown and design rationale. |
| [`docs/wire-format.md`](docs/wire-format.md) | On-wire / on-disk format (schema v5). |
| [`docs/api-stability.md`](docs/api-stability.md) | Per-module stability classification. |
| [`docs/compat-matrix.md`](docs/compat-matrix.md) | Runtime / engine / GPU compatibility. |
| [`docs/consistency.md`](docs/consistency.md) | Strong / quorum / eventual semantics. |
| [`docs/security.md`](docs/security.md) | Authn / authz / SSRF / encryption overview. |
| [`docs/release.md`](docs/release.md) | Versioning + release process. |
| [`docs/deployment.md`](docs/deployment.md) | docker-compose / k8s install paths. |
| [`docs/operations/slo.md`](docs/operations/slo.md) | Latency / availability targets. |
| [`docs/operations/upgrade.md`](docs/operations/upgrade.md) | Rolling upgrade + rollback. |
| [`docs/operations/backup-restore.md`](docs/operations/backup-restore.md) | Dual-store backup + disaster recovery. |
| [`docs/operations/incident-response.md`](docs/operations/incident-response.md) | On-call runbook. |
| [`docs/operations/capacity.md`](docs/operations/capacity.md) | Capacity planning. |
| [`docs/faq.md`](docs/faq.md) | Frequently asked questions. |

Release history is in [`CHANGELOG.md`](CHANGELOG.md); the
v3.0.0 breaking-change summary lives at the top of that file.

The OpenAPI spec is generated at runtime — start a server with
`membrane serve` and visit `/openapi.json`, or run:

```bash
membrane config --openapi > docs/openapi.json
```

---

## Build

```bash
python -m build
```

---

## Release

See [`docs/release.md`](docs/release.md) — version is bumped
in `pyproject.toml`, the changelog updated, a `vX.Y.Z` tag is
cut, and the release workflow publishes the sdist / wheel to
PyPI and the image to `ghcr.io/sachncs/membrane`.

---

## Architecture

Membrane separates the KV cache from GPU memory and treats it as a
distributed, content-addressed memory fabric:

- **Fragment data model** — A KV cache is decomposed into
  content-addressable :class:`Fragment` objects keyed by hash with a
  :class:`PayloadIdentity` describing layer/token span, dtype,
  and shape.
- **Indices** — Four specialized in-memory indices (exact, semantic,
  positional, co-access) over the same fragment set, exposed through a
  single :class:`Index` facade.
- **Reconstruction** — The :class:`Reconstructor` walks the
  indices to assemble a context, falling back to prefill when coverage
  is incomplete.
- **Routing** — Three coordinated routers (`Economic`, `Latency`,
  `Joint` in `membrane.analytical`) pick the best node for each
  request based on access history and live telemetry.
- **Cluster management** — :class:`Server` runs bootstrap,
  heartbeat, failure-detection, gossip, and replication loops in
  background threads (via `membrane.network.cluster.Cluster`).

### Mathematical Guarantees

1. **Six analytical equations** — Eqs. (1)–(6) from the paper are
   reproduced verbatim in `model/throughput_model.py` with the
   Table-5 fit.
2. **Content-addressing** — Two fragments with the same hash are
   byte-identical, enabling deduplication across tenants.
3. **Consistent hashing** — Adding or removing a node moves only
   `O(K/N)` keys.
4. **Bounded rank** — Fragment graphs are sparse; co-access and
   subgraph retrieval use bounded-depth BFS.
5. **TTL eviction** — Expired fragments are evicted before any LRU
   pass, guaranteeing no stale read of post-TTL content.

See [docs/architecture.md](docs/architecture.md) for full design
rationale and extension points.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |
| HTTP server | [FastAPI](https://fastapi.tiangolo.com/) / [uvicorn](https://www.uvicorn.org/) / stdlib `http.server` |
| gRPC | [grpcio](https://grpc.io/docs/languages/python/) + grpcio-tools |
| Persistence | [Redis](https://redis.io/) |
| Compute | [PyTorch](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), OpenAI/Anthropic APIs |
| Lint/Format | [ruff](https://docs.astral.sh/ruff/) |
| Type Check | [mypy](https://mypy-lang.org/) (strict) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Load Balancer | nginx |

---

## Roadmap

- **v3.0.x** — Current series: encrypted blob store at rest,
  deny-by-default mTLS, per-route scope checks, typed cluster
  errors, the v5 wire format, and the Python 3.10-3.13 support
  matrix.
- **v3.1** — Authn/authz split: Vault-backed secret rotation
  and OIDC federation.
- **v4.0** — Stable cluster protocol freeze; multi-region
  replication policies.
- **v5.0** — Native speculative-decode integration; second
  major wire break.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Pull request process
- Coding standards
- Test expectations

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin
