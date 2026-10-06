# API stability

What you can depend on across Membrane releases. The public API follows [Semantic Versioning](https://semver.org/), with the guarantees below, so integrations built on the stable surface keep working until the next major version.

Stability statements refer to the public surface exported by `membrane.__all__`, the package root. Deep paths such as `membrane.network.*` and `membrane.compute.*` are covered by the [per-module table](#per-module-stability-table), which marks each path as stable, experimental, or internal.

> [!IMPORTANT]
> Membrane is classified as alpha until 4.0. Read the **Breaking** section of the [CHANGELOG](../CHANGELOG.md) before each upgrade.

## Stable API surface (from v3.0.0)

The following names are **stable** from v3.0.0 onward:

* Memory objects: `Fragment`,
  `Prefix`,
  `Segment`,
  `Artifact`,
  `Trace`, and the discriminator
  `FragmentKind`.
* Stable fragment identity: `PayloadIdentity`
  (the durable ten-field fingerprint; this supersedes the removed
  `membrane.types.Signature` alias, which was deleted in 0.2.0).
* Serving plane: `Node`,
  `Origin`,
  `Replica`.
* Placement: `Ring`,
  `Shard`.
* Reconstruction: `Reconstructor`.
* Transfer: `TransferService`.
* Persistence: the `PersistenceBackend`
  protocol plus the
  `Memory`,
  `Redis`, and
  `CachingPersistence`
  implementations.
* Compute: `Backend` plus the
  `CPU`,
  `GPU`,
  `Transformers`,
  `OpenAI`,
  `Anthropic`, and
  `Ollama` backends.
* Transports: `FastAPIServer`.
* Composition: `Server`.
* Auth: the `Authenticator` protocol
  (concrete `APIKeyAuthenticator` lives at
  `membrane.auth.apikey`, `MTLSAuthenticator` at
  `membrane.auth.mtls`; `MTLSConfig` at `membrane.transport.tls`).
* HTTP API: the routes, request bodies, status codes, and route
  scopes documented in [Security](security.md) and served by
  `membrane serve`.
* Client: `MembraneClient`,
  `AsyncMembraneClient`, and the
  `MembraneClientError` hierarchy.
* CLI: `membrane` subcommands, their flags, and the
  `MEMBRANE_*` environment variables.
* Errors: the `Error` hierarchy
  (`NetworkError`, `SchemaError`, `MigrationError`, etc.).
* Logging: `configure_logging`.
* Wire format: the JSON envelope produced by
  `to_dict` / `from_dict`
  (schema version 5; see `docs/wire-format.md`).
* Prometheus metric names and label sets exposed by
  `membrane.metrics` and the FastAPI middleware.

## Per-module stability table

| Module path | Status | Notes |
|-------------|--------|-------|
| `membrane.*` (root exports) | **stable** (3.0+) | Mirrors `membrane.__all__`. |
| `membrane.network.cluster` | **stable** (3.0+) | Cluster lifecycle; covered by chaos tests. |
| `membrane.network.peer` | **stable** (3.0+) | Peer-to-peer HTTP client. |
| `membrane.network.gossip` | **stable** (3.0+) | Gossip payload + merge. |
| `membrane.network.membership` | **stable** (3.0+) | Membership table + heartbeat loop. |
| `membrane.network.config` | **stable** (3.0+) | `ClusterConfig` + `validate_config`. |
| `membrane.compute.openai` | **stable** (3.0+) | |
| `membrane.compute.anthropic` | **stable** (3.0+) | |
| `membrane.compute.ollama` | **stable** (3.0+) | |
| `membrane.compute.transformers` | **stable** (3.0+) | |
| `membrane.compute.cpu` / `gpu` | **stable** (3.0+) | |
| `membrane.compute.remote` | **stable** (3.0+) | Shared HTTP-LLM base class. |
| `membrane.transport.fastapi` | **stable** (3.0+) | |
| `membrane.transport.ops` | **experimental** | Per-op surface is being stabilised; helpers may move. |
| `membrane.transport.tls` | **stable** (3.0+) | mTLS configuration and SSL contexts. |
| `membrane.client` | **stable** (3.0+) | Sync and async HTTP clients. |
| `membrane.persistence.redis` | **stable** (3.0+) | |
| `membrane.persistence.memory` | **stable** (3.0+) | |
| `membrane.persistence.cache` | **stable** (3.0+) | |
| `membrane.security.encryption` | **stable** (3.0+) | AES-256-GCM + `DecryptError`. |
| `membrane.security.url_allowlist` | **stable** (3.0+) | SSRF guard. |
| `membrane.security.tenant` | **stable** (3.0+) | `TenantAuthorizer`. |
| `membrane.security.key_rotation` | **stable** (3.0+) | |
| `membrane.serialization` | **stable** (3.0+) | Wire-format envelope + `SCHEMA_VERSION`. |
| `membrane.audit` | **stable** (3.0+) | Append-only audit log. |
| `membrane.transport.authz` | **stable** (3.0+) | Per-route scope check. |
| `membrane.secrets` | **stable** (3.0+) | Backend-agnostic secrets vault. |
| `membrane.wire.v3` | **experimental** | Generated protobuf / gRPC stubs; not used by `membrane serve`. |
| `membrane.otel_tracer` | **experimental** | Subject to minor-version renames. |
| `membrane.disagg` | **experimental** | Prefill/decode disaggregation (REST + gRPC, `[disagg]` extra). |
| `membrane.adapters` | **stable** (3.0+) | Engine integration adapters. |
| `membrane.quantization` | **stable** (2.0+) | K/V tensor quantization. |
| `membrane.metrics` | **stable** (3.0+) | Prometheus collectors. |
| `membrane.resilience` | **stable** (3.0+) | Retry / circuit-breaker policies. |
| `membrane.cli` | **stable** (3.0+) | Typer + Rich CLI. |
| `membrane.analytical` | **stable** (3.0+) | Decision / policy classes. |
| `membrane.canonical` | **stable** (3.0+) | Canonical byte framing. |
| `membrane.compat` | **stable** (3.0+) | `ModelCompatibilityFingerprint`. |
| `membrane.errors` | **stable** (3.0+) | Typed exception hierarchy. |

Internal modules (prefixed `membrane._*`) are **not** covered
by this promise and may move without notice.

## Deprecation policy

* A deprecation warning is emitted in `n.x` for any breaking
  change planned for `n+1.0`.
* Deprecations are documented in `CHANGELOG.md` with a
  removal target.
* Deprecations are removed no sooner than the minor version
  after their introduction.

## Wire format versioning

The on-wire JSON format carries a `schema_version`
field managed by
`membrane.serialization.SCHEMA_VERSION` (= 5). Bumping
the version is a breaking change and requires a major version
bump. See `docs/wire-format.md` for the full spec.

## References

* `CHANGELOG.md` — release history
* `docs/release.md` — release process
* `docs/wire-format.md` — on-wire / on-disk format
* `docs/compat-matrix.md` — runtime / engine compat
