# Frequently Asked Questions

## General

### What is Membrane?

A distributed, content-addressed KV-cache fabric for LLM serving. It
separates the KV cache from GPU memory and shares it across a cluster,
so prefill work done once can be reused by other requests, tenants,
and regions.

### What is the relationship to the paper?

Membrane implements the analytical model of:

> **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter**
> Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang
> arXiv:2604.15039v2

Equations (1)–(6) and the Section 4 case study are reproduced
(`python scripts/demo.py`). The runtime around them (fragments,
indices, reconstruction, clustering, auth) is this project's own.

### How mature is it?

Membrane is versioned 3.x and classified as **alpha**. The HTTP API,
authentication, single-node serving, and local / Kubernetes clustering
are tested end to end in CI; see the
[CHANGELOG](../CHANGELOG.md) for what changed recently. Expect
breaking changes between minor versions until 4.0, and read
[API stability](api-stability.md) before depending on internals.

### Can I `pip install membrane`?

No. The `membrane` name on PyPI belongs to an unrelated project.
Install from source as shown in the [Quickstart](getting-started.md),
or install straight from Git:

```bash
pip install "membrane[server] @ git+https://github.com/sachncs/membrane.git"
```

Tagged releases also attach a wheel to each
[GitHub release](https://github.com/sachncs/membrane/releases) and
publish a container image to `ghcr.io/sachncs/membrane`.

## Setup

### Which Python versions are supported?

Python 3.14 only. Membrane uses 3.14 features (deferred annotations,
`compression.zstd`, template strings, subinterpreters, `uuid7`) and
declares `requires-python = ">=3.14,<3.15"`. `uv sync` installs 3.14
for you.

### Do I need Redis?

No. Nodes keep state in memory by default. Pass `--redis
redis://host:6379/0` (or set `MEMBRANE_REDIS_URL`) to persist node
state in Redis.

### Do I need a GPU?

No. The default `cpu` compute backend simulates prefill, which is
enough to run the API, clustering, and examples. Real KV extraction
needs the `gpu` or `transformers` backend and their extras.

### Why does `membrane serve --host 0.0.0.0` refuse to start?

A node reachable from the network must authenticate callers. Configure
`--api-key-file` or mTLS (see [Security](security.md)), or pass
`--allow-unauthenticated` for a throwaway development setup.

## Using it

### Why does `/store` return 422?

The fragment's `payload_ref` points at bytes that are not in the node's
content store. KV bytes reach the store out of band (through an engine
adapter, a shared store, or `/prefill`); `/store` publishes metadata
for bytes that already exist. A fragment with `payload_ref: null` is
metadata-only and is always accepted.

### Why does a `strong` write return 503?

It could not reach `--quorum-count` copies (local plus peers) before
the timeout, so it was rolled back. Check `membrane cluster-status`,
or lower `--quorum-count` / use `--consistency eventual`. See
[Consistency levels](consistency.md).

### Why can't my peers reach each other in Kubernetes?

Peer calls to private addresses are blocked unless they are seeds or in
`MEMBRANE_PEER_NETWORKS`. Set it to your pod CIDR. See
[Deployment](deployment.md#8-kubernetes).

### Why does `retrieve` say `found: false` for a fragment I stored as another key?

Reads are tenant-scoped. A key whose subject is `acme` cannot read
`globex` fragments, and the node answers exactly as if the fragment did
not exist.

## Development

### How do I run the tests?

```bash
pip install -e ".[dev]"
pytest tests/
ruff check membrane tests && ruff format --check membrane tests
mypy membrane
```

### How do I add a compute backend?

1. Subclass `Backend` from `membrane/compute/base.py` and implement
   `prefill`, `generate`, `available`, and `device_name`.
2. If the backend cannot write real KV bytes, leave
   `simulated_payload` as is; otherwise write the bytes to a
   `ContentStore` and override it to return `None`.
3. Register a factory in `COMPUTE_BACKENDS` in `membrane/server.py`.
4. Add tests under `tests/membrane/compute/`.

## Troubleshooting

### `cannot reach Membrane at http://localhost:8080`

No node is listening there. Start one with `membrane serve --daemon`,
or point the client at the right URL (`--base-url`, or `MEMBRANE_URL`
for the examples).

### `401 unauthorized` / `403 forbidden`

`401`: the key is missing or unknown. `403`: the key is valid but lacks
the route's scope (for example a `read` key calling `/store`).
