# Frequently Asked Questions

## General

### What is Membrane?

Membrane is a **Global Contextual Memory Fabric** for LLM inference. It provides a distributed, content-addressed memory system that enables cross-datacenter LLM serving by separating the KV cache from GPU memory and distributing it across a cluster.

### What is the relationship to the paper?

This repository is a reproduction and extension of:

> **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter**  
> Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang  
> arXiv:2604.15039v2

The analytical throughput model (Equations 1–6) and case-study baselines are reproduced verbatim. The codebase extends the paper with fragment data models, multiple indices, reconstruction engine, and multi-node networking.

### Is this production-ready?

Membrane is in **alpha** (version 0.1.x). The core data model and analytical model are stable. The server and networking components are functional but may change. See [Deployment Guide](deployment.md) for production considerations.

## Setup & Installation

### What Python versions are supported?

Python 3.10, 3.11, 3.12, and 3.13.

### Do I need Redis?

No. Redis is optional and only required for the production persistence backend. For development and testing, the in-memory backend is the default.

### Do I need a GPU?

No. Membrane includes a CPU compute backend by default. GPU support (via PyTorch CUDA) and remote LLM API support (OpenAI, Anthropic, Ollama) are optional extras.

## Development

### How do I run tests?

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### How do I run type checking?

```bash
python -m mypy membrane/
```

### How do I add a new compute backend?

1. Create a new file in `membrane/compute/` (e.g., `my_backend.py`)
2. Implement the `ComputeBackend` protocol from `membrane/compute/backend.py`
3. Register it in the CLI's backend selection logic
4. Add tests in `tests/membrane/compute/`

## Deployment

### How do I deploy with Docker?

See the [Deployment Guide](deployment.md#3-docker-deployment).

### How do I scale horizontally?

Membrane supports multi-node operation. Use the `ClusterManager` and consistent hashing to distribute fragments across nodes. For Kubernetes, consider StatefulSets with persistent volumes.

### How do I configure Redis?

Set the `MEMBRANE_REDIS_URL` environment variable:

```bash
export MEMBRANE_REDIS_URL=redis://:password@redis-host:6379/0
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'membrane'"

Make sure you've installed the package in editable mode:

```bash
pip install -e ".[dev]"
```

### Tests fail with import errors

Ensure your Python version is 3.10+ and you've activated your virtual environment.

### Docker build fails

Check that Docker and Docker Compose are installed and running. Run `docker compose build --no-cache` to force a clean build.
