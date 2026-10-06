# Memory API and routing

Ask a node what an inference engine needs to know before it runs a prompt: what is already cached, where it lives, and whether to compute the rest locally or offload it. These routes sit alongside store and retrieve, and every one authenticates and is tenant-scoped like the rest of the API.

## Routes

| Route | Scope | Answers |
|-------|-------|---------|
| `POST /reconstruct` | `read` (`write` with `prefill`) | The cached fragments covering a prompt, in token order |
| `GET`, `POST /prefix/lookup` | `read` | How many leading tokens of a prompt are cached on this node |
| `GET`, `DELETE /sessions/{id}` | `read`, `write` | The fragments a session has read |
| `POST /objects`, `GET /objects/{hash}` | `write`, `read` | Typed memory objects: prefixes, KV segments, documents, tool traces |
| `POST /route` | `read` | Where to fetch cached KV, where to prefill, where to store, and whether to offload |

The Python clients (`MembraneClient`, `AsyncMembraneClient`) have a method for each: `reconstruct`, `prefix_lookup`, `session`, `put_object`, `get_object`, and `route`. Fragments of other tenants are never visible; see [Security](security.md#tenant-isolation).

## Reconstruct a prompt

```bash
curl -s localhost:8080/reconstruct -H "Authorization: Bearer $KEY" \
  -d '{"tokens": [1, 2, 3, ...], "model_id": "llama-3-8b"}'
```

```json title="Response"
{
  "fragments": [{"identity": {"payload_hash": "…", "token_span": [0, 127], …}, …}],
  "coverage": 0.85,
  "missing": [[256, 299]],
  "prefilled": false,
  "prefetch": ["…"]
}
```

The node walks its indexes: the longest cached prefix first, then fragments adjacent to it, then any fragment that fills a gap.

- A fragment is returned only when its content hash matches the prompt tokens it covers. KV computed for a different prompt is never returned just because it sits at the same position or has a similar embedding.
- `missing` lists the token spans that are not cached, and `coverage` is the fraction that is.
- `prefetch` lists fragments that past requests read right after these.

With `"prefill": true`, the node computes the missing spans with its compute backend, stores them under the caller's tenant, and returns the complete set.

## Look up a cached prefix

```bash
curl -s "localhost:8080/prefix/lookup?model_id=llama-3-8b&tokens=1,2,3,4"
```

The response has `matched_tokens`, `total_tokens`, `full`, and the `fragments` covering the match. Answers are memoized, and an answer is forgotten as soon as one of its fragments leaves the node. For prompts longer than 4,096 tokens, `POST` the same fields as a JSON body.

## Sessions

A read that carries `X-Membrane-Session: <id>` (on `/retrieve` or `/reconstruct`) is recorded under that session, and `GET /sessions/<id>` returns the history, oldest first. Sessions are per tenant and bounded: 10,000 sessions of 1,000 reads each, evicting the least recently active first.

## Typed memory objects

`POST /objects` stores an object and the bytes behind it in one request. The node derives the content hash from the bytes, so a caller cannot bind a hash to content it does not hold.

| `kind` | Fields |
|--------|--------|
| `prefix` | `tokens` |
| `segment` | `layer`, `head`, `token_span`, `tensor_shape`, `data` (base64 KV bytes) |
| `artifact` | `source_url`, `data` (base64 document bytes), `token_count` |
| `trace` | `tool_name`, `input`, `output` |

Every kind may also carry `reuse_score` and `ttl`. `GET /objects/<hash>` returns `kind`, the object's fields, and `data`.

## Route a request

```bash
curl -s localhost:8080/route -d '{"tokens": [...], "model_id": "llama-3-8b", "local_cached_tokens": 0}'
```

```json title="Response"
{
  "policy": "latency",
  "placement": {"fetch_from": "node-2", "fetch_from_url": "https://node-2:8080",
                "store_on": "node-1", "prefill_on": "node-1", "reason": "nearest holder", …},
  "fragments": [{"content_hash": "…", "node_id": "node-1", "url": null},
                {"content_hash": "…", "node_id": "node-2", "url": "https://node-2:8080"}],
  "matched_tokens": 256,
  "reuse": true,
  "offload": {"target": "membrane", "incremental_length": 44, "cached_prefix_length": 256,
              "cross_cluster_cache_transfer": false, "threshold": 100}
}
```

| Field | Meaning |
|-------|---------|
| `fragments`, `matched_tokens` | The prompt's cached prefix across the cluster: this node's index first, then each further window looked up in the gossiped location directory |
| `reuse` | Whether fetching that KV beats recomputing it, from the peers' measured latency and the bytes involved |
| `offload` | With `--route-threshold N`: whether to prefill on the caller's own engine (`pd-p`) or offload to Membrane (`membrane`). The threshold rises while the node's queue is saturated and relaxes after; every ten minutes it is re-fitted to the prompt lengths seen |

A request may send `content_hash` instead of `tokens` to place a single fragment.

### Placement policies

`--placement` chooses the policy. Each is a `membrane.placement` plugin, so you can add your own; see [Plugins](plugins.md).

| Policy | Fetch from | Store on | Prefill on |
|--------|------------|----------|------------|
| `ring` (default) | A ring owner holding it, else the nearest holder | Ring primary | This node |
| `latency` | Lowest measured latency | Ring primary | This node |
| `selector` | Lowest load score (latency, GPU, memory, bandwidth) | Least-loaded owner | Least-loaded node |
| `economic` | Lowest latency | Highest value density minus cost | This node |
| `joint` | Lowest latency | Least memory pressure | Least GPU load and latency |

Peers report their load in every heartbeat: round-trip latency (a moving average), memory pressure, GPU load, region, and role. A peer in another `--region` costs bandwidth.

## Background policies

| Flag | Effect |
|------|--------|
| `--promote-replicas N` | Every 30 seconds, fragments read at least three times with a reuse score of 0.7 or more are copied, bytes included, to the least-loaded peers until they have `N` copies. Admins can change both thresholds at runtime with `POST /admin/policy` |
| `--dynamic-roles` | Every 30 seconds the node re-evaluates its role (`memory_host`, `prefill_worker`, `decode_worker`) from its own and its peers' load, and advertises it in its heartbeat |
| `--region NAME` | Advertised to peers, and used by replica placement and routing |
| `--origin HOST:PORT` | Run as a regional cache in front of an origin: a read that misses is fetched from the origin (bytes verified) and kept as a non-primary copy. Runs without `--peer`, and presents `--peer-api-key-file` to the origin |
| `--require-compat MODEL[:DTYPE]` | Refuse `POST /store` of fragments stamped for another model or tokenizer (`409`); fragments the node prefills are stamped |

> [!NOTE]
> Independently of these flags, `POST /store` refuses (`409`) a fragment whose content hash is already bound to a different identity, such as another layer range or token span, rather than aliasing two different KV tensors.
