# Architecture

Membrane separates the KV cache from GPU memory. KV segments become
immutable, content-addressed **fragments** held by a cluster of nodes;
serving engines look fragments up instead of recomputing prefill, and
only prefill the spans nobody holds.

```text
  clients / engine adapters (vLLM · SGLang · TensorRT-LLM)
                     │  HTTP or HTTPS (mTLS), bearer keys
                     ▼
 ┌──────────────────────── membrane serve ────────────────────────┐
 │  FastAPI routes ── authz (route scopes) ── ops (store/retrieve) │
 │        │                                        │               │
 │  compute backend (prefill)          Node: fragments + indices   │
 │                                     content store (KV bytes)    │
 │        │                                        │               │
 │  Cluster: membership · heartbeat · gossip · ring/shard · quorum │
 └──────────────────────────────┬──────────────────────────────────┘
                                │ peer HTTP(S) with credentials
                     other nodes (same process layout)
```

## Request path

0. **Admit.** ASGI middleware assigns an `X-Request-ID` (carried by
   every log line), applies the per-credential rate limit (`429`), and
   bounds in-flight requests (`503` + `Retry-After` when saturated;
   `membrane/transport/limits.py`).
1. **Authenticate.** Every route except `/livez` and `/readyz` runs
   `enforce_route_scope` (`membrane/transport/authz.py`) against the
   configured API-key or mTLS authenticator, then checks the route's
   scope (`read`, `write`, `admin`).
2. **Operate.** Route handlers in `membrane/transport/routes_fastapi.py`
   delegate to transport-agnostic functions in
   `membrane/transport/ops.py` and `ops_cluster.py`.
3. **Store.** `op_store` checks that the fragment's payload bytes are in
   the node's content store, applies the tenant check, writes locally,
   and for `strong` / `quorum` writes fans out to replicas in parallel
   and waits for acknowledgements (`QuorumReplicator`,
   `membrane/quorum.py`). It returns as soon as quorum is reached or the
   deadline passes, and the remaining budget caps every peer retry. A
   failed quorum rolls back the local write and returns `503`.
4. **Retrieve.** `op_retrieve` returns the fragment's metadata only if
   the caller's tenant may read it and its bytes are readable;
   undecryptable bytes are reported as `corrupt`.
5. **Prefill.** `op_prefill` runs the compute backend over the prompt
   and stores the resulting fragments as primaries.

## Components

### Server runtime (`membrane/runtime/`)

`membrane.server.Server` wires the node, compute backend, persistence,
cluster, and HTTP transport together. The pieces live in
`membrane/runtime/`:

| Module | Role |
|--------|------|
| `settings.py` | `ServerSettings` (frozen, validated) and `build_server`, which enforces the startup policy (authentication beyond loopback, private secret files, admin peer key) |
| `plugins.py` | `PluginRegistry` for compute backends, authenticators, and content stores; extended through entry points ([Plugins](plugins.md)) |
| `components.py` | Builders for persistence, the content store, authentication, and peer access |
| `persistence_writer.py` | Ordered, bounded write-behind queue that keeps Redis off the node lock, retries through outages, and flushes on shutdown |
| `lifecycle.py` | `PeriodicTask` (checkpoint, sweeper) and `run_until_signalled`, which turns `SIGTERM` into a drain |
| `observability.py` | Dashboard events (ring buffer) and diagnostics |

`membrane serve` parses flags into `ServerSettings`, calls
`build_server`, starts the server, and waits for a signal.

### Memory objects

| Class | Module | Role |
|-------|--------|------|
| `PayloadIdentity` | `membrane/identity.py` | Ten-field content address: model, tokenizer, layer / head / token ranges, dtype, shape, payload hash |
| `Fragment` | `membrane/fragment.py` | Immutable KV segment: identity, `payload_ref`, size, TTL, reuse score, tenant, consistency |
| `Prefix`, `Segment`, `Artifact`, `Trace` | `membrane/*.py` | Higher-level memory objects built from fragments |

### Node and storage

| Class | Module | Role |
|-------|--------|------|
| `Node` | `membrane/node.py` | Holds fragments in memory; TTL expiry, weighted-LRU and graph-aware eviction, tenant checks |
| `Index` | `membrane/index.py` | Facade over exact, semantic, positional, and co-access indices |
| `ContentStore` implementations | `membrane/content_store.py` | KV bytes: `InProcessBytes`, `FilesystemBlob` (AES-256-GCM); `EncryptedInProcessBytes` in `content_store_encrypted.py` |
| `Memory`, `Redis`, `CachingPersistence` | `membrane/persistence/` | Persistence backends for node state, written behind by `PersistenceWriter` |
| `Sweeper`, `TombstoneTable` | `membrane/gc.py` | Periodic TTL sweep and soft-delete propagation |

### Compute backends (`membrane/compute/`)

All subclass `Backend` (`base.py`) and are selected by name from the
compute plugin registry with `membrane serve --compute`:

| Name | Class | Notes |
|------|-------|-------|
| `cpu` | `CPU` | Deterministic simulator; default |
| `gpu` | `GPU` | PyTorch CUDA (`[gpu]` extra) |
| `transformers` | `Transformers` | HuggingFace models (`[local-llm]` extra) |
| `openai`, `anthropic`, `ollama` | `OpenAI`, `Anthropic`, `Ollama` | Remote APIs; KV bytes are simulated because the APIs do not expose them |

`KVBackend` (`kv.py`) extracts real KV tensors from a local model and
writes them to a `ContentStore`.

### Cluster (`membrane/network/`)

| Class | Role |
|-------|------|
| `Cluster` | Owns the subsystems below and the background threads; restarts a loop that raises |
| `Membership` | Peer table; seed bootstrap with retry |
| `Heartbeat`, `ThresholdDetector` | Liveness and failure detection |
| `Gossip`, `GossipState` | Membership, fragment locations, Bloom / Merkle inventory digests (cached on large inventories), tombstones |
| `Ring`, `Shard` | Consistent-hash placement of primaries and replicas |
| `Peer`, `PeerCredentials` | Outbound HTTP(S) client; bearer key or mTLS client cert |
| `Replicator` | Keeps primaries replicated: new primaries each sweep, a digest-based full pass every `repair_interval_sec` |

Outbound peer URLs pass the SSRF guard
(`membrane/security/url_allowlist.py`): seed hosts and the configured
`--peer-network` CIDRs are allowed; other private addresses are not.

### Routing and the analytical model

| Module | Contents |
|--------|----------|
| `membrane/model/throughput.py` | Eqs. (1)–(6) of the Prefill-as-a-Service paper |
| `membrane/model/optimizer.py` | Grid search over routing threshold and PD split, parallel across subinterpreters (`InterpreterPoolExecutor`) |
| `membrane/model/scheduler.py` | `DualTimescaleScheduler`: short-term routing, long-term reallocation |
| `membrane/model/simulator.py` | Case-study simulation (`python scripts/demo.py`) |
| `membrane/latency.py`, `economic.py`, `joint.py` | Latency, cost, and joint placement routers |

### Engine adapters (`membrane/adapters/`)

`MembraneVLLMAdapter`, `MembraneSGLangAdapter`, and
`MembraneTrtAdapter` connect serving engines' KV pools to a Membrane
cluster. Each ships an in-memory client for tests.

## Design principles

1. **Content-addressed.** Identical KV segments have identical
   identities, so work is shared across requests, tenants, and regions.
2. **Immutable.** Fragments are never modified; new versions get new
   identities.
3. **Reconstruction-driven.** Context is rebuilt from fragments, and
   only uncovered spans are prefilled.
4. **Secure by default.** Public binds require authentication; reads
   are tenant-scoped; peers authenticate to each other.
5. **Fail closed.** Strong writes that cannot reach quorum are rolled
   back and reported, never silently downgraded.
