# FAQ

Answers to the questions teams ask most when evaluating, deploying, and developing Membrane. If yours is not here, [open an issue](https://github.com/sachncs/membrane/issues).

## Evaluating Membrane

### What is Membrane?

Membrane is a shared KV-cache for LLM serving. It stores the KV cache produced by prefill outside GPU memory, addresses it by content, and serves it to every node in a cluster. A prompt that has been prefilled once, by any node, does not need to be prefilled again.

### What problem does it solve?

Prefill is the expensive, compute-bound part of serving a long prompt. Fleets recompute the same system prompts, documents, and conversation histories on every node that receives them. Membrane turns that repeated work into cache hits, which lowers time-to-first-token and frees GPU capacity.

### How does it relate to the Prefill-as-a-Service paper?

Membrane implements the analytical model from:

> **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter**<br>
> Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang. arXiv:2604.15039v2

Equations (1)–(6) and the Section 4 case study are reproduced by `python scripts/demo.py`. The runtime around them (fragments, indices, reconstruction, clustering, security) is this project's own.

### How mature is it?

Membrane is at version 3.x and classified as **alpha**. The HTTP API, authentication, single-node serving, and clustering on Kubernetes are tested end to end in CI on every change, including rolling restarts, node loss, and scale-out. Expect breaking changes between minor versions until 4.0. Read [API stability](api-stability.md) before you depend on internals, and the [CHANGELOG](../CHANGELOG.md) before each upgrade.

### What license is it under?

MIT. You can use, modify, and redistribute it, including commercially.

## Installation

### Can I `pip install membrane`?

Not from PyPI: the `membrane` name there belongs to an unrelated project. Install from source as in the [Quickstart](getting-started.md), or directly from Git:

```bash
pip install "membrane[server] @ git+https://github.com/sachncs/membrane.git"
```

Each tagged release also attaches a wheel to its [GitHub release](https://github.com/sachncs/membrane/releases) and publishes a container image to `ghcr.io/sachncs/membrane`.

### Which Python versions are supported?

Python 3.14 only (`requires-python = ">=3.14,<3.15"`). Membrane relies on 3.14 features such as the standard-library `compression.zstd` module and free-threading. `uv sync` installs 3.14 for you.

### Do I need Redis?

No. A node keeps its state in memory by default. Pass `--redis redis://host:6379/0` (or set `MEMBRANE_REDIS_URL`) to persist node state so it survives restarts. Redis Sentinel and Redis Cluster are supported for high availability; see [Deployment](deployment.md).

### Do I need a GPU?

Not to run Membrane. The default `cpu` compute backend simulates prefill, which is enough for the API, clustering, and examples. Extracting real KV tensors needs a GPU-backed engine (vLLM, SGLang, TensorRT-LLM) or the `gpu` / `transformers` backends.

## Running in production

### Why does `membrane serve --host 0.0.0.0` refuse to start?

A node reachable from the network must authenticate its callers. Configure `--api-key-file` or mTLS (see [Security](security.md)). For a throwaway development setup only, `--allow-unauthenticated` overrides the check.

### Why can't my peers reach each other on Kubernetes?

Peer calls to private addresses are blocked unless the address is a seed or falls inside `MEMBRANE_PEER_NETWORKS`. Set it to your pod CIDR, as described in [Deployment](deployment.md#kubernetes).

### Why does a `strong` write return 503?

The write could not reach `--quorum-count` copies (the local copy plus peers) before its deadline, so it was rolled back. Check `membrane cluster-status`. If you can accept weaker guarantees, lower `--quorum-count` or use `--consistency eventual`; see [Consistency levels](consistency.md).

### Why does `/store` return 422?

The fragment's `payload_ref` points at bytes that are not in the node's content store. KV bytes reach the store first (through an engine adapter, a shared store, or `/prefill`), and `/store` then publishes the metadata for them. A fragment with `payload_ref: null` is metadata-only and is always accepted.

### How are tenants isolated?

Every fragment belongs to a tenant, and each tenant keeps its own copy of the content it stores, so two tenants that cache the same prompt both get cache hits. A key can read only its own tenant's fragments and the shared `public` tenant's. A read of another tenant's fragment is indistinguishable from a miss. See [Security](security.md#identical-content-across-tenants).

## Developing

### How do I run the tests and checks?

```bash
uv sync --frozen --extra dev
pytest tests/
ruff check membrane tests && ruff format --check membrane tests
mypy membrane
```

CI also enforces coverage (at least 84% in total and 70% per module), naming and docstring rules, and cluster tests on [kind](https://kind.sigs.k8s.io/). [CONTRIBUTING](../CONTRIBUTING.md) has the full list.

### How do I add a compute backend, authenticator, or store?

Write a plugin. Implement the interface, then register a factory under the matching Python entry point (for example `membrane.compute`); Membrane discovers it at startup and you select it with a flag such as `--compute NAME`. No fork is needed. [Plugins](plugins.md) lists every extension point with examples.

## Troubleshooting

### `cannot reach Membrane at http://localhost:8080`

No node is listening at that address. Start one with `membrane serve --daemon`, or point the client at the right URL with `--base-url` (or `MEMBRANE_URL` for the examples).

### `401 unauthorized` or `403 forbidden`

`401` means the key is missing or unknown. `403` means the key is valid but lacks the route's scope, for example a `read` key calling `/store`. The scope of every route is listed in [Security](security.md#authorization).
