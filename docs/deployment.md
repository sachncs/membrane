# Deployment Guide

## Overview

Membrane can be deployed as a **library**, **simulator**, or **containerized service**. This guide covers all deployment options.

## 1. Library Installation

```bash
# Clone and install in editable mode
git clone https://github.com/sachncs/membrane.git
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

## 4. Docker Compose (Multi-Node)

```bash
docker compose up --build
```

This starts:
- **Redis** (port 6379) — persistence backend
- **Membrane** (port 8080) — API server
- **nginx** (port 80) — reverse proxy

### Profiles

```bash
# Run tests only
docker compose --profile test run --rm membrane-tests
```

## 5. Systemd Service (Linux)

```bash
sudo cp deployment/membrane.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now membrane
sudo journalctl -u membrane -f
```

## 6. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMBRANE_LOG_LEVEL` | `INFO` | Logging level |
| `MEMBRANE_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MEMBRANE_NODE_ID` | `membrane-0` | Unique node identifier |
| `MEMBRANE_TRANSPORT` | `http` | Transport protocol (`http` or `grpc`) |
| `MEMBRANE_COMPUTE` | `cpu` | Compute backend |
| `MEMBRANE_PORT` | `8080` | Server listen port |
| `MEMBRANE_HOST` | `0.0.0.0` | Server bind host |

## 7. PyPI Package

```bash
pip install build twine
python -m build
python -m twine upload dist/*
```

## 8. Production Considerations

Membrane ships with several production-ready components:

- **Persistent storage** — Redis backend with LRU eviction
- **Network transports** — HTTP (stdlib or FastAPI), gRPC, and peer-to-peer gossip
- **Compute backends** — CPU, GPU (PyTorch CUDA), and remote LLM APIs
- **Monitoring** — `/metrics` and `/heartbeat` endpoints; TUI dashboard

For hardened production deployments, consider:

- **Prometheus / Grafana** metrics exporter
- **Kubernetes StatefulSets** for horizontal scaling
- **TLS / mTLS** on gRPC and HTTP transports
- **Rate limiting and authorization** on the public API surface
- **S3-compatible blob storage** for large fragment offload
