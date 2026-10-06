# Prefill / decode disaggregation

Prefill (compute-bound) and decode (memory-bound) run best on different
machines. Membrane nodes can split them: a prefill node computes a
prompt's KV cache and keeps it; a decode node pulls that KV from the
prefill node and generates, without recomputing the prompt.

```bash
membrane serve --role prefill --peer decode-1:8080 ...   # prefill nodes
membrane serve --role decode  --peer prefill-1:8080 ...  # decode nodes
```

`--role both` (the default) serves both phases. A node's role is in its
heartbeat, so peers know where prefilled KV lives.

## REST

| Route | Scope | Does |
|-------|-------|------|
| `POST /disagg/prefill` | `write` | Reuse cached KV for the prompt, prefill the rest, return a `kv_handle` |
| `POST /disagg/prefill/batch` | `write` | The same for `{"requests": [...]}` |
| `POST /disagg/decode` | `write` | Generate from a `kv_handle` |
| `GET /disagg/healthz` | public | Liveness |

```bash
curl -s localhost:8080/disagg/prefill -H "Authorization: Bearer $KEY" \
  -d '{"request_id": "r1", "model_id": "llama-3-8b", "token_ids": [1, 2, 3]}'
# {"request_id": "r1", "kv_handle": "9f2c…", "prefill_ms": 3.1, "prompt_len": 3, "cached_prefix_len": 0}

curl -s decode-1:8080/disagg/decode -H "Authorization: Bearer $KEY" \
  -d '{"request_id": "r1", "kv_handle": "9f2c…", "model_id": "llama-3-8b", "max_tokens": 64}'
# {"request_id": "r1", "token_ids": [...], "finished": true}
```

The prefill node stores the KV fragments and a manifest naming them under
the handle. The decode node finds the manifest (locally, through the
gossiped location directory, or by asking prefill-capable peers), copies
any fragment it lacks from the prefill node with its bytes verified (as a
non-primary copy), and runs its compute backend.

| Status | Meaning |
|--------|---------|
| 404 | No reachable node knows the handle for this tenant, or its KV expired |
| 409 | This node's role does not serve the phase |
| 400 | Malformed request, or the handle was prefilled for another model |

Handles are per tenant: the same prompt prefilled by two tenants yields
the same handle, but each tenant can only decode its own.

## gRPC

`--grpc-port N` serves the same calls (`Prefill`, `BatchPrefill`,
`Decode`) over gRPC (`membrane/disagg/transfer.proto`). It authenticates
like HTTP: a bearer key in the `authorization` metadata, or the verified
client certificate when the node runs mTLS (the listener uses the node's
certificate). Status codes: `UNAUTHENTICATED`, `PERMISSION_DENIED`,
`NOT_FOUND`, `FAILED_PRECONDITION` (role), `INVALID_ARGUMENT`.

gRPC needs `membrane[disagg]` (in the image: `docker build --build-arg
EXTRAS=disagg .`). grpcio has no free-threaded build, so on a
free-threaded Python use the REST routes.

## Engine adapters

The vLLM, SGLang, and TensorRT-LLM adapters keep KV in a Membrane node
through `membrane.adapters.remote`, which stores named KV bundles
(`PUT`/`GET`/`HEAD /kv/{handle}?model_id=`, `write` / `read` scope, per
tenant):

```python
from membrane.adapters.remote import HTTPClusterClient, HTTPSGLangClient, HTTPTrtClient, connect
from membrane.adapters.vllm import MembraneVLLMConnector

client = connect("https://membrane-0:8080", api_key=KEY)
connector = MembraneVLLMConnector(client=HTTPClusterClient(client), n_layers=32, model_id="llama-3-8b")
```

The vLLM connector saves each request's blocks of every layer under the
prompt's handle; another engine instance that sees the same prompt loads
them instead of recomputing.
