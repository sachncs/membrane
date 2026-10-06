# Prefill and decode

Run prefill and decode on different machines. Prefill is compute-bound and decode is memory-bound, so they run best on different hardware. A prefill node computes a prompt's KV cache and keeps it; a decode node pulls that KV and generates, without recomputing the prompt.

## Assign roles

```bash title="Split a fleet by role"
membrane serve --role prefill --peer decode-1:8080 ...   # prefill nodes
membrane serve --role decode  --peer prefill-1:8080 ...  # decode nodes
```

`--role both`, the default, serves both phases. Each node advertises its role in its heartbeat, so peers know where prefilled KV lives.

## REST API

| Route | Scope | Purpose |
|-------|-------|---------|
| `POST /disagg/prefill` | `write` | Reuse cached KV for the prompt, prefill the rest, and return a `kv_handle` |
| `POST /disagg/prefill/batch` | `write` | The same for `{"requests": [...]}` |
| `POST /disagg/decode` | `write` | Generate from a `kv_handle` |
| `GET /disagg/healthz` | public | Liveness |

```bash title="Prefill on one node, decode on another"
curl -s localhost:8080/disagg/prefill -H "Authorization: Bearer $KEY" \
  -d '{"request_id": "r1", "model_id": "llama-3-8b", "token_ids": [1, 2, 3]}'
# {"request_id": "r1", "kv_handle": "9f2c…", "prefill_ms": 3.1, "prompt_len": 3, "cached_prefix_len": 0}

curl -s decode-1:8080/disagg/decode -H "Authorization: Bearer $KEY" \
  -d '{"request_id": "r1", "kv_handle": "9f2c…", "model_id": "llama-3-8b", "max_tokens": 64}'
# {"request_id": "r1", "token_ids": [...], "finished": true}
```

### How a handle is resolved

1. The prefill node stores the KV fragments, plus a manifest naming them, under the handle.
2. The decode node finds the manifest: locally, through the gossiped location directory, or by asking prefill-capable peers.
3. It copies any fragment it lacks from the prefill node, with the bytes verified, as a non-primary copy.
4. It runs its compute backend from the restored KV.

### Errors

| Status | Meaning |
|--------|---------|
| `400` | Malformed request, or the handle was prefilled for another model |
| `404` | No reachable node knows the handle for this tenant, or its KV expired |
| `409` | This node's role does not serve the requested phase |

> [!NOTE]
> Handles are per tenant. The same prompt prefilled by two tenants yields the same handle, but each tenant can decode only its own.

## gRPC

`--grpc-port N` serves the same calls (`Prefill`, `BatchPrefill`, `Decode`) over gRPC, defined in `membrane/disagg/transfer.proto`. It authenticates like HTTP: a bearer key in the `authorization` metadata, or the verified client certificate when the node runs mTLS.

| gRPC status | Meaning |
|-------------|---------|
| `UNAUTHENTICATED`, `PERMISSION_DENIED` | Missing credentials, or the wrong scope |
| `NOT_FOUND` | Unknown or expired handle |
| `FAILED_PRECONDITION` | This node's role does not serve the phase |
| `INVALID_ARGUMENT` | Malformed request |

gRPC needs `membrane[disagg]`; for the image, build with `docker build --build-arg EXTRAS=disagg .`.

> [!WARNING]
> grpcio has no free-threaded build. On free-threaded Python, use the REST routes.

## Serving engine adapters

The vLLM, SGLang, and TensorRT-LLM adapters keep KV in a Membrane node through `membrane.adapters.remote`, which stores named KV bundles (`PUT`/`GET`/`HEAD /kv/{handle}?model_id=`, with `write` and `read` scopes, per tenant):

```python title="Connect vLLM to Membrane"
from membrane.adapters.remote import HTTPClusterClient, connect
from membrane.adapters.vllm import MembraneVLLMConnector

client = connect("https://membrane-0:8080", api_key=KEY)
connector = MembraneVLLMConnector(client=HTTPClusterClient(client), n_layers=32, model_id="llama-3-8b")
```

The vLLM connector saves each request's blocks for every layer under the prompt's handle. Another engine instance that sees the same prompt loads them instead of recomputing.
