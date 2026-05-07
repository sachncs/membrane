# Deployment Guide

Membrane is a pure-Python library with zero runtime dependencies. It can be deployed as a **library**, **simulator**, or **containerized service**.

## 1. Library Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd membrane
pip install -e ".[dev]"

# Verify
python -c "import membrane; print(len(membrane.__all__), 'exports')"
```

## 2. Run Simulations Locally

```bash
# Paper reproduction demo
python scripts/demo.py

# Full Membrane multi-phase demo
python scripts/demo_full.py

# Multi-node simulation
python scripts/demo_membrane.py

# Run all tests
pytest tests/ -q
```

## 3. Docker Deployment

### Build
```bash
docker build -t membrane:latest .
```

### Run Paper Reproduction
```bash
docker run --rm membrane:latest python scripts/demo.py
```

### Run Tests
```bash
docker run --rm membrane:latest pytest tests/ -q
```

### Run Interactive Shell
```bash
docker run --rm -it membrane:latest python
```

## 4. Docker Compose (Multi-Node Simulation)

```bash
docker compose up --build
```

This runs `scripts/demo_membrane.py` inside a container with logging output.

## 5. Systemd Service (Linux)

Copy `deployment/membrane.service` to `/etc/systemd/system/` and enable:

```bash
sudo cp deployment/membrane.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now membrane
sudo journalctl -u membrane -f
```

The service runs the multi-node simulation on boot and logs to journald.

## 6. CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/ci.yml`) is provided:

- **Triggers**: Push to `master`, pull requests
- **Jobs**:
  1. Type-check with `mypy`
  2. Run 548 tests on Python 3.10–3.13
  3. Build Docker image
  4. Run demos inside container

## 7. PyPI Package

Build and upload:

```bash
pip install build twine
python -m build
python -m twine upload dist/*
```

The `pyproject.toml` already defines the package metadata with zero runtime dependencies.

## 8. Production Considerations

Membrane ships with several production-ready components already in place:

1. **Persistent storage backend** — Redis backend (`persistence/redis_backend.py`) with LRU eviction
2. **Network transports** — HTTP (stdlib or FastAPI), gRPC, and peer-to-peer gossip (`network/`)
3. **Compute backends** — CPU, GPU (PyTorch CUDA), and remote LLM APIs (OpenAI, Anthropic, Ollama)
4. **Monitoring** — `/metrics` and `/heartbeat` endpoints; TUI dashboard via `membrane dashboard`

For a hardened production deployment, additional work is recommended:

- **Prometheus / Grafana** metrics exporter beyond the basic endpoints
- **Kubernetes StatefulSets** for horizontal scaling of Membrane nodes
- **TLS / mTLS** on gRPC and HTTP transports
- **Rate limiting and authZ** on the public API surface
- **S3-compatible blob storage** for large fragment offload

These are outside the scope of the paper reproduction but are natural extension points.
