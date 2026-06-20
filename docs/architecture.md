# Architecture

## System Overview

Membrane is a distributed memory fabric designed for LLM inference serving. It separates the KV cache from GPU memory and distributes it across a cluster of storage/compute nodes.

```
                    ┌─────────────────────────────┐
                    │        Load Balancer         │
                    │          (nginx)              │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │     Membrane Server          │
                    │  ┌────────┐  ┌───────────┐  │
                    │  │ HTTP   │  │ gRPC      │  │
                    │  │ Server │  │ Server    │  │
                    │  └────┬───┘  └─────┬─────┘  │
                    │       │            │         │
                    │  ┌────▼────────────▼─────┐  │
                    │  │   Reconstruction       │  │
                    │  │   Engine               │  │
                    │  └────────────┬───────────┘  │
                    │               │              │
                    │  ┌────────────▼───────────┐  │
                    │  │   Index System          │  │
                    │  │ ┌──────┐ ┌───────────┐ │  │
                    │  │ │Exact │ │ Semantic  │ │  │
                    │  │ │Index │ │ Index     │ │  │
                    │  │ ├──────┤ ├───────────┤ │  │
                    │  │ │Posit.│ │ Co-Access │ │  │
                    │  │ │Index │ │ Index     │ │  │
                    │  │ └──────┘ └───────────┘ │  │
                    │  └────────────┬───────────┘  │
                    │               │              │
                    │  ┌────────────▼───────────┐  │
                    │  │   Fragment Store        │  │
                    │  │ (Memory / Redis)        │  │
                    │  └────────────────────────┘  │
                    └─────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
     │  Node A       │ │  Node B      │ │  Node C      │
     │  (Origin)     │ │  (Replica)   │ │  (Replica)   │
     └───────────────┘ └──────────────┘ └──────────────┘
```

## Module Breakdown

### Model Layer (`membrane/model/`)

Analytical throughput model from the "Prefill-as-a-Service" paper:

- **ThroughputModel** — Equations (1)–(6) for per-instance and system throughput
- **Optimizer** — Grid search over routing threshold `t` and PD split ratio
- **Scheduler** — Dual-timescale scheduling (short-term routing, long-term reallocation)
- **Simulator** — End-to-end discrete-event simulation for baseline comparison

### Core Data Model

- **Fragment** — Immutable, content-addressed KV segment
- **KVSegment** — Low-level key-value tensor pair
- **StructuralSignature** — Model/layer/token span metadata
- **Prefix** — Prompt prefix with version tracking

### Compute Backends (`membrane/compute/`)

| Backend | Use Case |
|---------|----------|
| `CPUBackend` | Default, no GPU required |
| `GPUBackend` | PyTorch CUDA acceleration |
| `TransformersBackend` | HuggingFace model inference |
| `OpenAIBackend` | OpenAI API |
| `AnthropicBackend` | Anthropic API |
| `OllamaBackend` | Local Ollama server |

### Persistence (`membrane/persistence/`)

| Backend | Use Case |
|---------|----------|
| `InMemoryBackend` | Development, testing |
| `RedisBackend` | Production with LRU eviction |

### Transport (`membrane/transport/`)

| Transport | Use Case |
|-----------|----------|
| `HTTPServer` | Standard REST API (stdlib or FastAPI) |
| `GrpcServer` | High-performance gRPC |

### Network (`membrane/network/`)

- **ClusterManager** — Node discovery and health monitoring
- **GossipState** — Gossip-based state replication
- **PeerClient** — Inter-node communication
- **RemoteTransferService** — Cross-node fragment transfer

### Routing

- **LatencyRouter** — Latency-aware request routing
- **EconomicRouter** — Cost-aware routing
- **HashRing** — Consistent hashing for shard assignment

### Multi-Tenancy

- **CanonicalStore** — Deduplicated fragment storage
- **TenantIsolation** — Per-tenant policy enforcement
- **TenantPolicy** — Access and resource policies

## Data Flow

1. **Ingest**: A prompt is received by the server
2. **Fragment**: The `FragmentationEngine` splits the prompt into fragments
3. **Index**: Fragments are indexed in all four indices
4. **Store**: Fragments are persisted (memory or Redis)
5. **Reconstruct**: On future requests, `rebuild_context()` assembles context from fragments
6. **Transfer**: Fragments can be synced across nodes via the transfer plane
7. **Route**: The scheduler routes requests based on load and cost

## Design Principles

1. **Content-Addressed**: Fragments are identified by their content hash, enabling automatic deduplication.
2. **Immutable**: Fragments are never modified after creation; updates create new versions.
3. **Reconstruction-Driven**: Context is rebuilt from fragments, not stored monolithically.
4. **Multi-Tenant**: First-class tenant isolation with configurable policies.
5. **Pluggable**: Compute, storage, and transport backends are interchangeable.
