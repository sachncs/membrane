# Getting Started with Membrane

## Overview

Membrane is a **Global Contextual Memory Fabric** for LLM inference. It provides a distributed, content-addressed, reconstruction-driven memory system that enables cross-datacenter LLM serving with optimal throughput and latency.

## Quick Start

### 1. Install

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Verify

```bash
python -c "import membrane; print(f'{len(membrane.__all__)} exports available')"
pytest tests/ -q
```

### 3. Run Demos

```bash
# Paper reproduction — analytical throughput model
python scripts/demo.py

# Full Membrane multi-phase demo
python scripts/demo_full.py

# Multi-node simulation
python scripts/demo_membrane.py
```

## Core Concepts

### Fragments

A **Fragment** is an immutable, content-addressed KV segment. Each fragment has:

- A unique content hash
- Structural signatures (model, layer, token span)
- Metadata (creation time, size, tenant)

### Indices

Membrane maintains four in-memory indices for fast fragment lookup:

| Index | Purpose |
|-------|---------|
| **Exact** | Hash-based exact match lookup |
| **Semantic** | Embedding similarity search |
| **Positional** | Interval-based adjacency (token ranges) |
| **Co-access** | Co-access pattern graph adjacency |

### Reconstruction Engine

The `rebuild_context()` method reconstructs LLM context from stored fragments, falling back to prefill when fragments are unavailable.

### Global Directory

Multi-node fragment location resolution. Tracks which nodes own which fragments across the cluster.

### Transfer Plane

Delta-sync and chunked, resumable fragment transfer between nodes. Supports delta encoding for bandwidth-efficient updates.

## Next Steps

- Read the [Architecture Guide](architecture.md) for system design details.
- Read the [Deployment Guide](DEPLOY.md) for production deployment instructions.
- See the [FAQ](faq.md) for common questions.
