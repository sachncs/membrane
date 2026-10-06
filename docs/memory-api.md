# Memory API and routing

Besides storing and retrieving fragments, a node answers the questions an
inference engine asks before it runs a prompt: what is already cached,
where it lives, and whether to compute the rest locally or offload it.
Every route below authenticates and is scoped like the rest of the API
([Security](security.md)); fragments of other tenants are never visible.

| Route | Scope | Answers |
|-------|-------|---------|
| `POST /reconstruct` | `read` (`write` with `prefill`) | The cached fragments covering a prompt, in token order |
| `GET`/`POST /prefix/lookup` | `read` | How many leading tokens of a prompt are cached here |
| `GET /sessions/{id}`, `DELETE /sessions/{id}` | `read` / `write` | The fragments a session read |
| `POST /objects`, `GET /objects/{hash}` | `write` / `read` | Typed memory objects: prefixes, KV segments, documents, tool traces |
| `POST /route` | `read` | Where to fetch cached KV, where to prefill, where to store, and whether to offload |

The Python clients (`MembraneClient`, `AsyncMembraneClient`) have a
method for each: `reconstruct`, `prefix_lookup`, `session`, `put_object`,
`get_object`, and `route`.

## Reconstruct a prompt

```bash
curl -s localhost:8080/reconstruct -H "Authorization: Bearer $KEY" \
  -d '{"tokens": [1, 2, 3, ...], "model_id": "llama-3-8b"}'
```

```json
{
  "fragments": [{"identity": {"payload_hash": "…", "token_span": [0, 127], …}, …}],
  "coverage": 0.85,
  "missing": [[256, 299]],
  "prefilled": false,
  "prefetch": ["…"]
}
```

The node walks its indexes: the longest cached prefix first, then
fragments adjacent to it, then any fragment that fills a gap. A fragment
is returned only when its content hash matches the prompt tokens it
covers, so KV computed for a different prompt is never handed out
because it sits at the same position or has a similar embedding.
`prefetch` lists fragments that past requests read right after these.

With `"prefill": true`, the uncovered spans are computed by the node's
compute backend, stored (owned by the caller's tenant), and returned.

## Look up a cached prefix

```bash
curl -s "localhost:8080/prefix/lookup?model_id=llama-3-8b&tokens=1,2,3,4"
```

returns `matched_tokens`, `total_tokens`, `full`, and the `fragments`
covering the match. Answers are memoized; an entry is forgotten as soon
as one of its fragments leaves the node. `POST` takes the same fields as
a JSON body for prompts longer than 4,096 tokens.

## Sessions

A read that carries `X-Membrane-Session: <id>` (on `/retrieve` or
`/reconstruct`) is recorded under that session. `GET /sessions/<id>`
returns the history, oldest first. Sessions are per tenant and bounded
(10,000 sessions of 1,000 reads each; the least recently active go
first).

## Typed memory objects

`POST /objects` stores an object and the bytes behind it in one request.
The node derives the content hash from the bytes, so a caller cannot bind
a hash to content it does not hold.

| `kind` | Fields |
|--------|--------|
| `prefix` | `tokens` |
| `segment` | `layer`, `head`, `token_span`, `tensor_shape`, `data` (base64 KV bytes) |
| `artifact` | `source_url`, `data` (base64 document bytes), `token_count` |
| `trace` | `tool_name`, `input`, `output` |

Each may also carry `reuse_score` and `ttl`. `GET /objects/<hash>`
returns `kind`, the object's fields, and `data`.

## Route a request

```bash
curl -s localhost:8080/route -d '{"tokens": [...], "model_id": "llama-3-8b", "local_cached_tokens": 0}'
```

```json
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

* `fragments` and `matched_tokens` cover the prompt's cached prefix
  across the cluster: this node's index first, then each further window
  looked up in the gossiped location directory.
* `reuse` says whether fetching that KV beats recomputing it, from the
  peers' measured latency and the bytes involved.
* `offload` (with `--route-threshold N`) says whether to prefill on the
  caller's own engine (`pd-p`) or offload to Membrane (`membrane`). The
  threshold rises while the node's request queue is saturated and relaxes
  after; every ten minutes it is re-fitted to the prompt lengths seen.
* A request may send `content_hash` instead of `tokens` to place one
  fragment.

`--placement` picks the policy; each is a `membrane.placement` plugin
([Plugins](plugins.md)):

| Policy | Fetch from | Store on | Prefill on |
|--------|------------|----------|------------|
| `ring` (default) | a ring owner holding it, else the nearest holder | ring primary | this node |
| `latency` | lowest measured latency | ring primary | this node |
| `selector` | lowest load score (latency, GPU, memory, bandwidth) | least-loaded owner | least-loaded node |
| `economic` | lowest latency | highest value density minus cost | this node |
| `joint` | lowest latency | least memory pressure | least GPU load and latency |

Peers report their load in every heartbeat: round-trip latency (moving
average), memory pressure, GPU load, region, and role. A peer in another
`--region` costs bandwidth.

## Background policies

| Flag | Effect |
|------|--------|
| `--promote-replicas N` | Every 30 s, fragments read at least three times with a reuse score of 0.7 or more are copied, bytes included, to the least-loaded peers until they have `N` copies |
| `--dynamic-roles` | Every 30 s the node re-evaluates its role (`memory_host`, `prefill_worker`, `decode_worker`) from its own and its peers' load, and advertises it in its heartbeat |
| `--region NAME` | Advertised to peers; used by replica placement and routing |
| `--origin HOST:PORT` | Run as a regional cache in front of an origin: a read that misses is fetched from the origin (bytes verified) and kept as a non-primary copy. Runs without `--peer`; presents `--peer-api-key-file` to the origin |
| `--require-compat MODEL[:DTYPE]` | Refuse `POST /store` of fragments stamped for another model or tokenizer (409); fragments the node prefills are stamped |

Independently of these flags, `POST /store` refuses (409) a fragment
whose content hash is already bound to a different identity (another
layer range or token span), rather than aliasing two different KV
tensors.
