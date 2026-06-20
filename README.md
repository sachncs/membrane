# Membrane

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/sachn-cs/membrane/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/membrane/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/sachn-cs/membrane)](https://github.com/sachn-cs/membrane/stargazers)

**Global Contextual Memory Fabric for cross-datacenter LLM serving.**

Membrane is a distributed, content-addressed, reconstruction-driven memory system built on the analytical throughput model from the ["Prefill-as-a-Service"](https://arxiv.org/abs/2604.15039) paper. It separates the KV cache from GPU memory and distributes it across a cluster, enabling optimal throughput and latency for LLM inference.

---

## Features

- **Analytical Throughput Model** — Verbatim reproduction of Equations (1)–(6) from the paper
- **Throughput-Optimal Configuration** — Grid search over routing threshold and PD split ratio
- **Dual-Timescale Scheduling** — Bandwidth-aware short-term routing with long-term reallocation
- **Fragment Data Model** — Immutable, content-addressed KV segments with structural signatures
- **Four In-Memory Indices** — Exact, Semantic, Positional, and Co-access lookup
- **Reconstruction Engine** — Context rebuilding from fragments with prefill fallback
- **Multi-Node Networking** — Gossip-based cluster management, consistent hashing, peer transfer
- **Multi-Tenant Isolation** — Canonical store with per-tenant policies and deduplication
- **Pluggable Backends** — CPU, GPU, Transformers, OpenAI, Anthropic, Ollama compute backends
- **Multiple Transports** — HTTP (stdlib + FastAPI) and gRPC server options
- **Redis Persistence** — LRU eviction and distributed storage
- **CLI with TUI Dashboard** — Live monitoring, cluster status, and interactive setup wizard
- **548+ Tests** — Comprehensive test suite across Python 3.10–3.13

## Installation

```bash
git clone https://github.com/sachn-cs/membrane.git
cd membrane
python -m venv .venv
source .venv/bin/activate

# Core installation (typer + rich CLI)
pip install -e ".[dev]"

# Optional: Server dependencies (FastAPI, gRPC, Redis)
pip install -e ".[server]"

# Optional: GPU backend (PyTorch CUDA)
pip install -e ".[gpu]"

# Optional: Local LLM backend (HuggingFace Transformers)
pip install -e ".[local-llm]"
```

## Quick Start

```bash
# Verify installation
python -c "import membrane; print(f'{len(membrane.__all__)} exports available')"

# Run paper reproduction demo
python scripts/demo.py

# Run full multi-phase demo
python scripts/demo_full.py

# Run multi-node simulation
python scripts/demo_membrane.py

# Start the server
membrane serve --node-id n1 --port 8080 --transport http --compute cpu
```

## Usage

### CLI Commands

```bash
# Start a Membrane server
membrane serve --node-id n1 --port 8080 --transport http --compute cpu

# Open live TUI dashboard
membrane dashboard --host localhost --port 8080

# Show server status
membrane status

# Show current configuration
membrane config
```

### Python API

```python
import membrane

# Create a fragment store
from membrane.fragment_store import FragmentStore
store = FragmentStore()

# Create fragments
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

sig = StructuralSignature(model="llama-3", layer=0, token_span=(0, 128))
frag = Fragment(content=b"kv-data", signature=sig)

# Store and retrieve
store.put(frag)
retrieved = store.get(frag.content_hash)
```

### Docker

```bash
# Build and run
docker compose up --build

# Run tests
docker compose --profile test run --rm membrane-tests
```

## Configuration

Configuration is managed via environment variables. See [`.env.example`](.env.example) for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMBRANE_LOG_LEVEL` | `INFO` | Logging level |
| `MEMBRANE_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MEMBRANE_NODE_ID` | `membrane-0` | Unique node identifier |
| `MEMBRANE_TRANSPORT` | `http` | Transport protocol (`http` or `grpc`) |
| `MEMBRANE_COMPUTE` | `cpu` | Compute backend |
| `MEMBRANE_PORT` | `8080` | Server listen port |
| `MEMBRANE_HOST` | `0.0.0.0` | Server bind host |

## Project Structure

```
membrane/
├── model/                    # Analytical model (paper reproduction)
│   ├── throughput_model.py   # Equations (1)–(6)
│   ├── optimizer.py          # Grid search optimizer
│   ├── scheduler.py          # Dual-timescale scheduler
│   ├── workload.py           # Log-normal workload generator
│   └── simulator.py          # End-to-end simulations
├── compute/                  # Compute backends (CPU, GPU, API)
├── persistence/              # Storage backends (Memory, Redis)
├── transport/                # Network transports (HTTP, gRPC)
├── network/                  # Cluster management and peer networking
├── fragment.py               # Core fragment data model
├── indices.py                # Four in-memory index types
├── reconstruction_engine.py  # Context reconstruction from fragments
├── server.py                 # Unified production server
├── cli.py                    # CLI with TUI dashboard
└── ...

tests/                        # 548+ tests across Python 3.10–3.13
scripts/                      # Demo scripts
deployment/                   # Systemd, nginx configs
docs/                         # Documentation
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run type checking
python -m mypy membrane/

# Run linting
ruff check membrane/ tests/

# Run format check
ruff format --check membrane/ tests/

# Auto-format code
ruff format membrane/ tests/

# Run with coverage
pytest tests/ --cov=membrane --cov-report=term-missing
```

### Helper Scripts

```bash
# Full setup: install, type-check, and run tests
bash scripts/setup.sh

# Clean build artifacts and caches
bash scripts/cleanup.sh
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| CLI | Typer + Rich |
| HTTP Server | FastAPI / uvicorn / stdlib |
| gRPC | grpcio / grpcio-tools |
| Persistence | Redis |
| Compute | PyTorch, HuggingFace Transformers, OpenAI/Anthropic APIs |
| Testing | pytest, mypy |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Load Balancer | nginx |

## Roadmap

- [ ] Kubernetes operator for automatic scaling
- [ ] Prometheus/Grafana metrics exporter
- [ ] TLS/mTLS for transport encryption
- [ ] Rate limiting and API key authentication
- [ ] S3-compatible blob storage backend
- [ ] WebAssembly compute backend
- [ ] Web UI dashboard
- [ ] gRPC streaming for real-time updates
- [ ] Fragment compression and deduplication improvements
- [ ] Multi-region replication policies

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
