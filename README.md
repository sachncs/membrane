<p align="center">
  <h1 align="center">Membrane</h1>
  <p align="center">Distributed, content-addressed KV-cache sharing for LLM serving clusters.</p>
  <p align="center">
    <a href="https://github.com/sachncs/membrane/actions/workflows/ci.yml"><img src="https://github.com/sachncs/membrane/actions/workflows/ci.yml/badge.svg?branch=master" alt="CI"></a>
    <img src="https://img.shields.io/badge/python-3.14-blue" alt="Python 3.14">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
    <a href="https://sachncs.github.io/membrane/"><img src="https://img.shields.io/badge/docs-sachncs.github.io%2Fmembrane-7c3aed" alt="Documentation"></a>
  </p>
</p>

Membrane separates the KV cache from GPU memory. KV segments become
immutable, content-addressed **fragments** held by a cluster of nodes,
so prefill work done once can be reused by other requests, tenants,
and regions; only the spans nobody holds are prefilled again.

It implements the analytical model of
[*Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*](https://arxiv.org/abs/2604.15039)
(Eqs. 1–6 and the Section 4 case study, reproducible with
`python scripts/demo.py`) and the runtime around it.

**Documentation: <https://sachncs.github.io/membrane/>**

## Quickstart

Membrane is installed from source (the `membrane` package on PyPI is an
unrelated project) and runs on Python 3.14 only.
[uv](https://docs.astral.sh/uv/) installs 3.14 and the locked
dependencies:

```bash
git clone https://github.com/sachncs/membrane.git
cd membrane
uv sync --frozen --extra server && source .venv/bin/activate

membrane serve --daemon &
membrane client prefill --prompt-tokens "1 2 3 4 5 6 7 8"
membrane client inventory
membrane client retrieve --hash b5ca3ba695d8ec8ce47bf6e7a2b579d0
```

The [Quickstart](docs/getting-started.md) continues with the Python
client, authentication, a three-node cluster, and the container image.

## Features

- **Content-addressed fragments.** A ten-field `PayloadIdentity`
  (model, tokenizer, layer / head / token ranges, dtype, shape, hash)
  makes identical KV segments findable across the cluster.
- **Four indices, one facade.** Exact, semantic, positional, and
  co-access lookups; the `Reconstructor` rebuilds a context and
  prefills only the gaps.
- **Clustering.** Gossip membership, heartbeat failure detection,
  consistent-hash placement, and strong writes that wait for replicas
  (KV bytes included) and fail closed when they cannot get them.
  Primaries hand off with verification on drain and rebalance when
  nodes join; repair pages only the inventory buckets that changed.
- **Memory API and routing.** `/reconstruct` assembles the cached KV for
  a prompt, `/prefix/lookup` reports how much is cached, and `/route`
  says where to fetch, prefill, and store, with pluggable placement
  policies ([Memory API](docs/memory-api.md)).
- **Prefill / decode disaggregation.** `--role prefill|decode`: decode
  nodes pull the KV a prefill node computed, over REST or gRPC
  ([Prefill / decode](docs/disaggregation.md)).
- **Scales up and out.** On free-threaded Python 3.14 (`Dockerfile.ft`)
  one node serves requests on several cores; adding nodes spreads
  ownership, and per-node consistency costs follow what changed.
- **Secure by default.** A node will not listen on a public address
  without API keys or mTLS; keyfiles hold only SHA-256 digests and must
  be private; every route checks a scope; reads are isolated per
  tenant; peers authenticate to each other.
- **Reliable under load.** Bounded concurrency with `503` +
  `Retry-After` backpressure, per-key rate limits, quorum writes that
  honour their deadline, and a `SIGTERM` drain that lets a 3-node
  cluster roll without a failed write.
- **Durable when you want it.** `--redis` persists fragment metadata
  (a single server, Sentinel, or Redis Cluster) and `--data-dir` keeps
  KV bytes on disk (AES-256-GCM, rotatable keys), so nodes survive
  restarts. An encrypted warm tier and KV quantization stretch memory.
- **Pluggable.** CPU simulator, PyTorch GPU, HuggingFace Transformers,
  OpenAI, Anthropic, Ollama, and KV-extraction compute backends;
  third-party compute backends, authenticators, content stores,
  persistence, eviction, secret providers, placement policies, and event
  hooks through entry points ([Plugins](docs/plugins.md)); adapters for
  vLLM, SGLang, and TensorRT-LLM that keep their KV in Membrane.
- **Operable.** JSON logs with request IDs, OpenTelemetry traces,
  Prometheus metrics with per-endpoint and per-tenant labels, a
  persistent hash-chained audit log, certificate hot reload, ACME and
  SPIFFE, a TUI dashboard, admin commands, hardened Python 3.14 container
  images, and Docker Compose, Kubernetes, and systemd configurations.

## Running it

| Goal | Where to look |
|------|---------------|
| Try it locally | [Quickstart](docs/getting-started.md) |
| Container, Compose, Kubernetes, systemd | [Deployment](docs/deployment.md) |
| API keys, mTLS, scopes, tenants | [Security](docs/security.md) |
| Write consistency and quorum | [Consistency levels](docs/consistency.md) |
| Reconstruction, prefix lookup, routing | [Memory API & routing](docs/memory-api.md) |
| Prefill / decode nodes, engine adapters | [Prefill / decode](docs/disaggregation.md) |
| Monitoring and alerts | [SLOs](docs/operations/slo.md) |
| Sizing | [Capacity planning](docs/operations/capacity.md) |
| Backups, upgrades, incidents | [Backup & restore](docs/operations/backup-restore.md), [Upgrades](docs/operations/upgrade.md), [Incident response](docs/operations/incident-response.md) |
| How it fits together | [Architecture](docs/architecture.md) |
| Extending it | [Plugins](docs/plugins.md) |
| Common questions | [FAQ](docs/faq.md) |

Every `membrane serve` flag has a `MEMBRANE_*` environment variable;
`membrane serve --help` lists them all.

## Development

```bash
uv sync --frozen --extra dev && source .venv/bin/activate
pytest tests/
ruff check membrane tests scripts examples
ruff format --check membrane tests scripts examples
mypy membrane
python tools/check_naming.py && python tools/check_docstrings.py
```

CI runs these on Python 3.14 from the lockfile and again on
free-threaded 3.14t, plus:
- Redis integration, including Sentinel and Redis Cluster;
- stress, chaos, and benchmark smoke tests;
- security scans (pip-audit, bandit, gitleaks, Trivy);
- smoke tests of both container images;
- a 5-node Kubernetes (kind) end-to-end test: rolling restart, primary
  loss, and scale-out;
- the documentation site build and a docs link check. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and
[docs/release.md](docs/release.md) for releases.

The site in [`site/`](site/) (Astro) publishes the landing page and
renders `docs/` to <https://sachncs.github.io/membrane/>.

## Status

Membrane is **alpha**: versioned 3.x, tested end to end, and expected
to change between minor versions until 4.0. See the
[CHANGELOG](CHANGELOG.md) and [API stability](docs/api-stability.md).

## Contributing, conduct, security

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
