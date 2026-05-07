# Membrane — Global Contextual Memory Fabric

This repository contains the Membrane system: a distributed, content-addressed,
reconstruction-driven memory fabric for LLM inference, built on top of the
analytical throughput model from:

> **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go
> Cross-Datacenter**  
> Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu,
> Weimin Zheng, Mingxing Zhang  
> arXiv:2604.15039v2

## What is reproduced

1. **Analytical throughput model** (Section 3.4.1, Equations 1–6)
   - Per-instance KV throughput `Phi_kv(l)`
   - Stage throughputs `Theta_membrane`, `Theta_pd-p`, `Theta_pd-d`
   - End-to-end system throughput `Lambda_max`

2. **Throughput-optimal configuration** (Section 3.4.2)
   - Grid search over routing threshold `t` and PD prefill/decode split
   - Optimality conditions from Equations 7–8

3. **Dual-timescale scheduling** (Section 3.4.3)
   - Short-term: bandwidth- and cache-aware routing with congestion response
   - Long-term: traffic-driven reallocation and re-optimization

4. **Workload generator** (Section 4.1)
   - Truncated log-normal request lengths (`mu=9.90`, `sigma=1.00`, `[128,128K]`)
   - Fixed output length of 1024 tokens

5. **Case-study baselines** (Section 4)
   - Membrane-PD (selective offloading)
   - Homogeneous PD
   - Naive Heterogeneous PD

6. **Evaluation metrics**
   - `Lambda_max` (sustainable throughput)
   - Mean and P90 TTFT
   - Cross-datacenter bandwidth utilization

## Membrane extensions

Beyond the paper reproduction, this codebase implements:

- **Fragment data model**: Immutable, content-addressed KV segments with structural signatures.
- **Four in-memory indices**: Exact, Semantic, Positional, and Co-access.
- **Graph layer**: Fragment relationship graph with weighted edges.
- **Reconstruction engine**: Reads fragments via `rebuild_context()` with fallback to prefill.
- **Global directory**: Multi-node fragment location resolution.
- **Transfer plane**: Delta-sync and chunked, resumable fragment transfer.
- **Multi-tenant deduplication**: Canonical store with tenant isolation.

## Setup

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

Core dependencies are `typer` and `rich` for the CLI. Optional extras are available:

- `pip install -e ".[server]"` — FastAPI, uvicorn, gRPC, Redis
- `pip install -e ".[gpu]"` — PyTorch CUDA backend
- `pip install -e ".[local-llm]"` — HuggingFace Transformers backend

## Running tests

```bash
pytest tests/ -v
```

## Running demos

```bash
# Paper reproduction demo
python scripts/demo.py

# Full Membrane multi-phase demo
python scripts/demo_full.py

# Multi-node simulation demo
python scripts/demo_membrane.py
```

## Project structure

```
membrane/
├── model/                    # Analytical model and simulator (paper reproduction)
│   ├── __init__.py
│   ├── profiler.py          # Table 5 profiling data + interpolation
│   ├── throughput_model.py  # Equations (1)–(6)
│   ├── optimizer.py         # Grid search for t and N_p/N_d
│   ├── workload.py          # Truncated log-normal generator
│   ├── router.py            # Length-based and cache-aware routing
│   ├── scheduler.py         # Dual-timescale scheduler
│   ├── metrics.py           # TTFT and bandwidth metrics
│   └── simulator.py         # End-to-end baseline simulations
├── compute/                 # Compute backends
│   ├── backend.py
│   ├── cpu_backend.py
│   ├── gpu_backend.py
│   ├── transformers_backend.py
│   ├── openai_backend.py
│   ├── anthropic_backend.py
│   └── ollama_backend.py
├── persistence/             # Storage backends
│   ├── memory_backend.py
│   └── redis_backend.py
├── transport/               # Network transports
│   ├── http_server.py       # stdlib HTTP server
│   ├── fastapi_server.py    # FastAPI + uvicorn server
│   └── grpc_server.py       # gRPC server
├── network/                 # Cluster and peer networking
│   ├── cluster_manager.py
│   ├── config.py
│   ├── gossip_state.py
│   ├── peer_client.py
│   └── remote_transfer.py
├── __init__.py              # Public API exports
├── server.py                # Unified production server
├── cli.py                   # Command-line interface + dashboard
├── fragment.py              # Core data model
├── exact_index.py           # Hash-based exact index
├── semantic_index.py        # Embedding similarity index
├── positional_index.py      # Interval-based adjacency index
├── co_access_index.py       # Co-access graph adjacency index
├── graph_manager.py         # Graph lifecycle management
├── fragment_graph.py        # Fragment relationship graph
├── reconstruction_engine.py # Context reconstruction from fragments
├── prefill_adapter.py       # Adapter to analytical model
├── global_directory.py      # Fragment location directory
├── distributed_directory.py # Shard-aware distributed directory
├── transfer_service.py      # Sender/receiver negotiation
├── chunked_transfer.py      # Chunk-based fragment transfer
├── delta_encoder.py         # Delta encoding for transport
├── canonical_store.py       # Multi-tenant deduplicated storage
├── tenant_isolation.py      # Tenant policy enforcement
├── kv_cache_manager.py      # Local KV cache management
├── membrane_node.py         # Node owning a shard
├── origin_node.py           # Primary fragment host
├── replica_node.py          # Replica fragment host
├── supernode.py             # Cluster coordination node
├── latency_router.py        # Latency-aware routing
├── economic_router.py       # Cost-aware routing
├── hash_ring.py             # Consistent hashing
├── workload_analyzer.py     # Session history analysis
├── session_tracker.py       # Per-session access tracking
├── node_telemetry.py        # Node health telemetry
├── dynamic_role_manager.py  # Compute/memory role assignment
├── joint_optimizer.py       # Joint placement optimization
├── offload_decision_engine.py # KV offload decisions
├── remote_prefill_dispatcher.py # Remote prefill dispatch
├── cluster_replicator.py    # Cluster-wide replication
├── promotion_policy.py      # Replica promotion logic
├── predictor.py             # Access pattern prediction
├── cost_model.py             # Cost estimation
├── value_density.py         # Value density scoring
├── semantic_cluster.py      # Semantic clustering
├── semantic_hash.py         # Semantic hashing utilities
├── structural_signature.py  # Model/layer/token span signatures
├── prefix.py                # Prefix data model
├── prefix_version_chain.py  # Version chain for prefixes
├── fragmentation_engine.py  # Prompt-to-fragment conversion
├── index_system.py          # Unified index facade
├── cache_metrics.py         # Cache performance metrics
├── tool_trace.py            # Tool execution traces
├── artifact.py              # Artifact data model
├── memory_object.py         # Memory object protocol
└── ...

tests/
├── test_*.py                # Model unit tests
└── membrane/
    ├── test_*.py            # Membrane unit and integration tests

scripts/
├── demo.py                 # Paper reproduction demo
├── demo_full.py            # Full Membrane multi-phase demo
└── demo_membrane.py        # Multi-node simulation demo
```

## Fidelity report

| Component | Status | Notes |
|-----------|--------|-------|
| Throughput equations (1)–(6) | **Exact** | Implemented verbatim from the paper. |
| Table 5 profiling data | **Exact** | All four measured points transcribed exactly. |
| Grid-search optimizer | **Exact** | Exhaustive 2-D search over `t` and `N_p`. |
| Workload distribution | **Exact** | Truncated log-normal with paper parameters. |
| Routing logic | **Exact** | Threshold and cache-aware rules from Sections 3.3 and 3.4.3. |
| Dual-timescale scheduler | **Approximate** | Congestion threshold value and relaxation rate are **ASSUMPTION** (not specified in paper). Long-term reallocation period is **NOT DETERMINED**. |
| Decode constants | **Approximate** | `T_decode = 25 ms` and `BS_max = 20` are **ASSUMPTION** inferred from Table 6 consistency, not explicitly stated. |
| Prefix cache model | **Approximate** | Simplified to per-request prefix lengths; hit-rate distribution is **NOT DETERMINED** by the paper. |
| TTFT model | **Approximate** | Adds KV transfer time to prefill time for Membrane requests; queuing delay is **NOT DETERMINED** (assumed negligible for steady-state throughput). |
| Interpolation | **Approximate** | Linear interpolation between measured profiling points; method is **NOT DETERMINED** by the paper. |
| Homogeneous baseline | **Exact** | Uses same total instance count, no Membrane. |
| Naive heterogeneous baseline | **Exact** | All prefill on Membrane, all decode on PD, no threshold. |

## Mismatch report

1. **Table 6 absolute numbers**: Because `Lambda_max` depends on conditional
   means `E[L | L > t]` and `E[L | L <= t]`, which are computed from a
   synthetic sample, exact reproduction of the paper's `t = 19.4K` and
   `Lambda_max = 3.24` requires either the exact same random seed or a very
   large sample. Our search with 20k–50k samples consistently finds `t` in
   the 16K–21K range and `Lambda_max` in the 3.0–3.4 range, which is
   statistically consistent with the paper.

2. **Decode SLO constants**: The paper states `BS_max` and `T_decode` are
   "SLO-governed constants" but does not give their values. We inferred
   `T_decode = 0.025 s` and `BS_max = 20` from the consistency of Table 6
   across the three baselines (see `optimizer.py` docstring for derivation).
   These are marked as **ASSUMPTION**.

3. **Scheduler parameters**: The paper describes the short-term scheduler's
   behavior conceptually ("raise the effective threshold") but does not
   specify exact utilization thresholds, step sizes, or the long-term
   reallocation period. We chose `congestion_threshold = 0.85` and a 10%
   threshold increase, with gradual 1% relaxation. These are marked as
   **ASSUMPTION**.

4. **Prefix cache evaluation**: The case study in Section 4 does not report
   prefix-cache hit rates. Our baseline reproduction assumes zero cache hits
   so that the results are driven solely by the routing threshold and
   hardware allocation, matching the paper's primary comparison. The router
   module still implements the cache-aware logic from Section 3.4.3 for
   extensibility.

5. **No GPU-level simulation**: The paper's system is a real serving stack.
   Our reproduction is an analytical/discrete-event simulator that captures
   the steady-state throughput model. Fine-grained GPU scheduling, TCP
   congestion control, and layer-wise pipelining (Section 3.3) are not
   modeled at the packet or kernel level.

## License

MIT
