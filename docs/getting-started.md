# Quickstart

Install Membrane, run a node, cache your first prompt, and grow to a three-node cluster with authentication. It takes about five minutes, and every command on this page runs in the test suite or CI.

**What you will have at the end:** a secured three-node cluster on your machine that stores prefill results once and serves them back by content hash.

> [!NOTE]
> Membrane is installed from source. The `membrane` package on PyPI is an unrelated project.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.14 | Required exactly; Membrane uses 3.14 features. [uv](https://docs.astral.sh/uv/) installs it for you. |
| `git` | To clone the repository. |
| Docker | Optional, for [step 7](#7-run-it-as-a-container). |

## 1. Install

uv reads `.python-version` and installs the exact dependency versions locked in `uv.lock`, so every machine gets the same build:

```bash title="Install with uv (recommended)"
git clone https://github.com/sachncs/membrane.git
cd membrane
uv sync --frozen --extra server
source .venv/bin/activate
membrane --version
```

On an existing Python 3.14, pip works too:

```bash title="Install with pip"
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[server]"
```

The `server` extra installs what `membrane serve` needs: FastAPI, uvicorn, the Redis client, and `cryptography`. Add others as you need them:

| Extra | Adds |
|-------|------|
| `dev` | Test, lint, and type-check tooling (`pytest`, `ruff`, `mypy`) |
| `transfer` | lz4 transfer compression and KV quantization (`numpy`, `lz4`; zstd is in the standard library) |
| `gpu`, `local-llm` | PyTorch and Hugging Face Transformers compute backends |
| `secrets-aws`, `secrets-gcp`, `secrets-vault` | Secret manager backends |
| `otel` | OpenTelemetry tracing |

Serving engines (vLLM, SGLang, TensorRT-LLM) are not extras. Install the engine in its own environment; `membrane.adapters` imports it only when used.

## 2. Start a node

```bash
membrane serve --daemon
```

```text title="Output"
2026-10-05 12:00:00,000 [INFO] membrane.cli: Membrane server started on 127.0.0.1:8080
  Node ID  : membrane-0
  Auth     : none (loopback only)
  Compute  : cpu
  Redis    : disabled (in-memory)
  ...
```

The node listens on `127.0.0.1:8080`. Leave it running and open a second terminal, or start it in the background with `&`. Without `--daemon`, the command opens a live dashboard in the terminal instead.

Confirm it is ready:

```bash
curl -s localhost:8080/readyz
# {"status":"ready"}
```

> [!TIP]
> Diagnostics go to stderr and command results to stdout, so `membrane client inventory | jq` works. Add `--log-format json` for one JSON object per log line.

## 3. Store and read fragments

A **fragment** is an immutable, content-addressed slice of KV cache. `prefill` runs the compute backend over a prompt and stores the resulting fragments on the node:

```bash
membrane client prefill --prompt-tokens "1 2 3 4 5 6 7 8"
membrane client inventory
```

```json title="Output"
{
  "digest": { "b5ca3ba695d8ec8ce47bf6e7a2b579d0": 1 },
  "node_id": "membrane-0"
}
```

Each key in `digest` is a fragment's content hash. Read one back:

```bash
membrane client retrieve --hash b5ca3ba695d8ec8ce47bf6e7a2b579d0
```

```json title="Output"
{ "found": true, "fragment": { "tenant_id": "public", "payload_size": 512, "ttl": 3600.0, "...": "..." } }
```

The same prompt always produces the same hash. That is how two requests, two nodes, or two regions find each other's prefill work instead of repeating it.

> [!NOTE]
> The default `cpu` backend is a simulator: it produces deterministic placeholder KV bytes, which is all you need to explore the API. Connect a real model with a [serving engine adapter or compute backend](plugins.md).

## 4. Use the Python client

```python title="quickstart.py"
import logging

from membrane.client import MembraneClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quickstart")

client = MembraneClient("http://localhost:8080")
result = client.prefill(list(range(256)), model_id="llama-3")
for frag in result["fragments"]:
    content_hash = frag["identity"]["payload_hash"]
    log.info("%s found=%s", content_hash, client.retrieve(content_hash)["found"])
client.close()
```

`AsyncMembraneClient` offers the same calls for `asyncio`. Errors are typed, so callers can handle each case:

| Exception | Raised when |
|-----------|-------------|
| `MembraneConnectionError` | The node is unreachable |
| `MembraneUnauthorizedError` | The key is missing, invalid, or lacks the scope (401, 403) |
| `MembraneServerError` | The node returned a 5xx response |

For a complete application, [`examples/rag_pipeline.py`](../examples/rag_pipeline.py) caches prompts for a retrieval-augmented generation service:

```bash
python examples/rag_pipeline.py
```

## 5. Turn on authentication

A node refuses to listen on any non-loopback address until authentication is configured:

```bash
membrane serve --host 0.0.0.0
# Refusing to serve unauthenticated on 0.0.0.0. Configure --api-key-file or mTLS ...
```

Generate a key for a tenant. The first line printed is the key itself, for the client. The second is its SHA-256 keyfile entry, for the server, which never stores the key:

```bash title="Create a key and serve with it"
membrane keys generate --subject acme --scope read --scope write > acme.txt
sed -n 2p acme.txt > api-keys && chmod 600 api-keys
membrane serve --host 0.0.0.0 --api-key-file api-keys --daemon
```

The key's subject (`acme`) is the tenant it reads and writes. Clients send the key with each request:

```bash
export MEMBRANE_API_KEY="$(sed -n 1p acme.txt)"
membrane client inventory --api-key "$MEMBRANE_API_KEY"
```

Every route except `/livez` and `/readyz` requires a key with the right scope: `read`, `write`, or `admin`.

> [!IMPORTANT]
> The server refuses to start with a keyfile that other users can read. Keep it `chmod 600`, owned by the user that runs Membrane.

For mTLS, SPIFFE, and the full scope table, see [Security](security.md).

## 6. Run a local cluster

Start three nodes, each seeded with the other two:

```bash title="Three local nodes"
membrane serve -n n1 -p 8080 --daemon --peer localhost:8081 --peer localhost:8082 &
membrane serve -n n2 -p 8081 --daemon --peer localhost:8080 --peer localhost:8082 &
membrane serve -n n3 -p 8082 --daemon --peer localhost:8080 --peer localhost:8081 &
membrane cluster-status --port 8080
```

Within a few seconds each node lists the other two as healthy. Fragments are placed on a consistent-hash ring and replicated with their bytes.

Writes use `strong` consistency by default: a store returns only after `--quorum-count` copies exist (2 by default, the local copy and one peer), and fails with `503` when too few peers are healthy. [Consistency levels](consistency.md) describes the alternatives.

To stop the cluster, run `kill %1 %2 %3` or `pkill -f "membrane serve"`. On `SIGTERM` a node drains gracefully: readiness and writes return `503`, it hands its primaries to peers, leaves the cluster, and exits within `--drain-timeout` (30 seconds by default).

## 7. Run it as a container

```bash title="Run the image"
docker build -t membrane .
docker run --read-only --tmpfs /tmp -p 8080:8080 \
  -v "$PWD/api-keys:/run/secrets/api-keys:ro" \
  -e MEMBRANE_API_KEY_FILE=/run/secrets/api-keys \
  membrane
```

The image runs Python 3.14 as a non-root user (uid 1000) on a read-only root filesystem and logs JSON. On Linux, give that user the keyfile: `sudo chown 1000 api-keys`. Every `membrane serve` flag has a `MEMBRANE_*` environment variable; see `membrane serve --help`.

Stop the container with `docker stop -t 40` so the drain can finish.

## Core concepts

| Concept | What it is |
|---------|------------|
| **Fragment** | An immutable KV segment addressed by a ten-field identity: model, tokenizer, layer, head, and token ranges, dtype, shape, and payload hash. |
| **Node** | A process that holds fragments in memory, with TTL expiry and weighted-LRU eviction. |
| **Index** | Exact, semantic, positional, and co-access lookups over a node's fragments. |
| **Reconstructor** | Rebuilds a prompt's context from fragments and prefills only the missing spans. |
| **Cluster** | Gossip membership, heartbeats, consistent-hash placement, and replication between nodes. |
| **Tenant** | Every fragment belongs to a tenant, and reads are authorized per tenant. |

## Next steps

- [Deployment](deployment.md): Docker Compose, Kubernetes, and systemd for production.
- [Security](security.md): API keys, mTLS, scopes, and tenant isolation.
- [Plugins](plugins.md): connect compute backends, authenticators, and storage without forking.
- [Architecture](architecture.md): how the pieces fit together.
- [FAQ](faq.md): common questions.
