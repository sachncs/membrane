# Quickstart

This walkthrough takes you from a fresh checkout to a running node,
your first cached fragments, and a three-node local cluster. It takes
about five minutes. Every command below is exercised by the test suite
or CI.

Membrane is installed from source; it is not published on PyPI (the
`membrane` package there is an unrelated project).

## Prerequisites

- Python 3.10 – 3.13
- `git`
- Optional: Docker, for the container image

## 1. Install

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server]"
membrane --version
```

`[server]` installs what `membrane serve` needs (FastAPI, uvicorn,
Redis client, `cryptography`). Other extras:

| Extra | Adds |
|-------|------|
| `dev` | Test, lint, and type-check tooling (`pytest`, `ruff`, `mypy`) |
| `transfer` | KV transfer engine and quantization (`numpy`, `lz4`, `zstandard`) |
| `gpu` / `local-llm` | PyTorch / HuggingFace Transformers compute backends |
| `vllm` / `sglang` / `trtllm` | Serving-engine adapters |
| `secrets-aws` / `secrets-gcp` / `secrets-vault` | Secret backends |
| `otel` | OpenTelemetry tracing |

## 2. Start a node

```bash
membrane serve --daemon
```

```text
Membrane server started on 127.0.0.1:8080
  Node ID : membrane-0
  Auth     : none (loopback only)
  Compute  : cpu
  Redis    : disabled (in-memory)
```

The node binds `127.0.0.1` by default. Leave it running and open a
second terminal (or run it in the background with `&`). Without
`--daemon` the command opens a live TUI dashboard instead.

Check it:

```bash
curl -s localhost:8080/readyz
# {"status":"ready"}
```

## 3. Store and read fragments

A **fragment** is an immutable, content-addressed slice of KV cache.
`prefill` runs the configured compute backend over a prompt and stores
the resulting fragments on the node. The default `cpu` backend is a
simulator: it produces deterministic placeholder KV bytes, which is all
you need to explore the API.

```bash
membrane client prefill --prompt-tokens "1 2 3 4 5 6 7 8"
membrane client inventory
```

```json
{
  "digest": { "b5ca3ba695d8ec8ce47bf6e7a2b579d0": 1 },
  "node_id": "membrane-0"
}
```

Each key in `digest` is a fragment's content hash. Retrieve one:

```bash
membrane client retrieve --hash b5ca3ba695d8ec8ce47bf6e7a2b579d0
```

```json
{ "found": true, "fragment": { "tenant_id": "public", "payload_size": 512, "ttl": 3600.0, "...": "..." } }
```

The same prompt always produces the same hash, which is how two
requests (or two tenants, or two regions) find each other's prefill
work.

## 4. Use the Python client

```python
from membrane.client import MembraneClient

client = MembraneClient("http://localhost:8080")
result = client.prefill(list(range(256)), model_id="llama-3")
for frag in result["fragments"]:
    content_hash = frag["identity"]["payload_hash"]
    print(content_hash, client.retrieve(content_hash)["found"])
client.close()
```

`AsyncMembraneClient` offers the same calls for `asyncio`. Errors are
typed: `MembraneConnectionError` (node unreachable),
`MembraneUnauthorizedError` (401 / 403), `MembraneServerError` (5xx).

A fuller example, [`examples/rag_pipeline.py`](../examples/rag_pipeline.py),
caches prompts for a RAG service:

```bash
python examples/rag_pipeline.py
```

## 5. Turn on authentication

A node refuses to listen on any non-loopback address until
authentication is configured:

```bash
membrane serve --host 0.0.0.0
# Refusing to serve unauthenticated on 0.0.0.0. Configure --api-key-file or mTLS ...
```

Create a keyfile (one `<key>:<subject>:<scopes>` per line; the subject
is the tenant the key reads and writes) and restart:

```bash
printf '%s:acme:read,write\n' "$(openssl rand -hex 32)" > api-keys
membrane serve --host 0.0.0.0 --api-key-file api-keys --daemon
```

Clients now pass the key:

```bash
export MEMBRANE_API_KEY="$(cut -d: -f1 api-keys)"
membrane client inventory --api-key "$MEMBRANE_API_KEY"
```

Every route except `/livez` and `/readyz` requires a key with the
right scope (`read`, `write`, or `admin`). See [Security](security.md)
for mTLS and the full scope table.

## 6. Run a local cluster

Start three nodes that seed from each other:

```bash
membrane serve -n n1 -p 8080 --daemon --peer localhost:8081 --peer localhost:8082 &
membrane serve -n n2 -p 8081 --daemon --peer localhost:8080 --peer localhost:8082 &
membrane serve -n n3 -p 8082 --daemon --peer localhost:8080 --peer localhost:8081 &
membrane cluster-status --port 8080
```

Each node lists the other two as healthy within a few seconds. Writes
default to `strong` consistency: a `store` returns only after
`--quorum-count` copies exist (2 by default: the local copy and one
peer), and fails with `503` when not enough peers are healthy. See
[Consistency levels](consistency.md).

Stop everything with `kill %1 %2 %3` (or `pkill -f "membrane serve"`);
nodes shut down gracefully on `SIGTERM`.

## 7. Run it as a container

```bash
docker build -t membrane .
docker run --read-only --tmpfs /tmp -p 8080:8080 \
  -v "$PWD:/run/secrets:ro" \
  -e MEMBRANE_API_KEY_FILE=/run/secrets/api-keys \
  membrane
```

The image is configured with `MEMBRANE_*` environment variables (every
`membrane serve` flag has one; see `membrane serve --help`).

## Core concepts

| Concept | What it is |
|---------|------------|
| **Fragment** | Immutable KV segment addressed by a ten-field `PayloadIdentity` (model, tokenizer, layer / head / token ranges, dtype, shape, payload hash). |
| **Node** | A process holding fragments in memory, with TTL expiry and weighted-LRU eviction. |
| **Index** | Exact, semantic, positional, and co-access lookups over the node's fragments. |
| **Reconstructor** | Rebuilds a prompt's context from fragments, prefilling only the missing spans. |
| **Cluster** | Gossip membership, heartbeats, consistent-hash placement, and replication between nodes. |
| **Tenant** | Every fragment carries a `tenant_id`; reads are authorized per tenant. |

## Next steps

- [Deployment](deployment.md): Docker Compose, Kubernetes, and systemd.
- [Security](security.md): API keys, mTLS, scopes, and tenant isolation.
- [Architecture](architecture.md): how the pieces fit together.
- [FAQ](faq.md).
- Reproduce the paper's case study: `python scripts/demo.py`.
