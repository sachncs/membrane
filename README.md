<p align="center">
  <h1 align="center">Membrane</h1>
  <p align="center">Global Contextual Memory Fabric for distributed, content-addressed KV-cache sharing across LLM serving clusters.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/membrane/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/membrane/ci.yml?branch=master" alt="CI"></a>
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
- **Multiple transports** — stdlib HTTP, FastAPI HTTP, and gRPC over the same logical surface
- **Redis persistence** — Optional durability with LRU eviction and inventory digests
- **CLI with TUI dashboard** — Live monitoring, cluster status, and interactive setup wizard

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
pip install -e ".[dev]"
```

### Optional extras

```bash
# Server dependencies (FastAPI, gRPC, Redis, httpx)
pip install -e ".[server]"

# GPU compute backend (PyTorch with CUDA)
pip install -e ".[gpu]"

# Local LLM backend (HuggingFace Transformers + tokenizers)
pip install -e ".[local-llm]"
```

**Requirements**: Python 3.10+ (3.10, 3.11, 3.12, 3.13 supported on CI)

---

## Quick Start

### Python API

```python
import membrane
from membrane.fragment import Fragment
from membrane.fragment_store import FragmentStore
from membrane.structural_signature import StructuralSignature

# Verify install and inspect the public surface.
print(f"{len(membrane.__all__)} exports available")

# Create a fragment store and store a fragment.
sig = StructuralSignature(model="llama-3", layer=0, token_span=(0, 128))
frag = Fragment(content=b"kv-data", signature=sig)

store = FragmentStore()
store.put(frag)
retrieved = store.get(frag.content_hash)
```

### CLI

```bash
# Reproduce the paper's analytical evaluation.
python scripts/demo.py
python scripts/demo_full.py

# Start a single-node server with the CPU backend.
membrane serve --node-id n1 --port 8080 --transport http --compute cpu

# Open a live TUI dashboard against a running server.
membrane dashboard --host localhost --port 8080

# Show cluster membership.
membrane cluster-status

# Show LLM-backend status.
membrane llm-status
```

### Docker

```bash
# Build and run a single-node server.
docker compose up --build

# Run the test suite inside the container.
docker compose --profile test run --rm membrane-tests
```

---

## Configuration

### Compute and transport

| Flag | Default | Description |
|------|---------|-------------|
| `--compute` | `cpu` | `cpu`, `gpu`, `ollama`, `openai`, `anthropic`, `transformers` |
| `--transport` | `http` | `http` (FastAPI), `stdlib` (stdlib HTTP), or `grpc` |
| `--redis` | _disabled_ | Redis URL when set, e.g. `redis://localhost:6379/0` |
| `--max-memory` | `1<<30` | Per-node memory budget in bytes |

### Cluster and replication

| Flag | Default | Description |
|------|---------|-------------|
| `--peer` | _none_ | Seed peer `host:port` (repeatable) |
| `--heartbeat-interval` | `2.0` | Heartbeat period in seconds |
| `--gossip-interval` | `5.0` | Gossip period in seconds |
| `--replica-count` | `2` | Replicas per primary shard |
| `--failure-remove-threshold` | `4` | Missed heartbeats before removing a peer |

### LLM backends

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-url` | _none_ | Base URL (used for Ollama or a custom OpenAI endpoint) |
| `--llm-model` | _none_ | Model identifier (e.g. `llama3.2`, `gpt-4o-mini`, `claude-3-sonnet`) |
| `--api-key` | _none_ | API key for OpenAI / Anthropic |

### Environment variables

See [`.env.example`](.env.example) for the full list. The most
commonly-used entries are:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMBRANE_LOG_LEVEL` | `INFO` | Logging level |
| `MEMBRANE_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MEMBRANE_NODE_ID` | `membrane-0` | Unique node identifier |
| `MEMBRANE_TRANSPORT` | `http` | Transport protocol |
| `MEMBRANE_COMPUTE` | `cpu` | Compute backend |
| `MEMBRANE_PORT` | `8080` | Server listen port |
| `MEMBRANE_HOST` | `0.0.0.0` | Server bind host |

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
│   ├── http_server.py                 # stdlib HTTP server
│   ├── fastapi_server.py              # FastAPI + uvicorn
│   └── grpc_server.py                 # gRPC servicer
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

## Architecture

Membrane separates the KV cache from GPU memory and treats it as a
distributed, content-addressed memory fabric:

- **Fragment data model** — A KV cache is decomposed into
  content-addressable :class:`Fragment` objects keyed by hash with a
  :class:`StructuralSignature` describing layer/token span.
- **Indices** — Four specialized in-memory indices (exact, semantic,
  positional, co-access) over the same fragment set, exposed through a
  single :class:`IndexSystem` facade.
- **Reconstruction** — The :class:`ReconstructionEngine` walks the
  indices to assemble a context, falling back to prefill when coverage
  is incomplete.
- **Routing** — Three coordinated routers (:class:`LatencyRouter`,
  :class:`EconomicRouter`, :class:`JointOptimizer`) pick the best node
  for each request based on access history and live telemetry.
- **Cluster management** — :class:`ClusterManager` runs bootstrap,
  heartbeat, failure-detection, gossip, and replication loops in
  background threads, sharing state with :class:`MembraneServer`.

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

- **v0.1.x** — Current series: paper reproduction, content-addressed fabric, multi-transport serving
- **v0.2.0** — TLS/mTLS for transport encryption, API key authentication
- **v0.3.0** — Prometheus/Grafana metrics exporter, Kubernetes operator for autoscaling
- **v1.0.0** — Stable API, gRPC streaming for real-time updates, multi-region replication policies

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