# API Stability Promise

Membrane's public API follows [Semantic Versioning](https://semver.org/)
with the additional guarantees below. Stability statements refer
to the public surface exported by ``membrane.__all__`` (the package
root). Deep paths (``membrane.network.*``, ``membrane.compute.*``,
etc.) are covered by the per-module stability table below; each
section names whether the path is stable, experimental, or
internal.

## Stable API surface (from v3.0.0)

The following names are **stable** from v3.0.0 onward:

* Memory objects: :class:`~membrane.fragment.Fragment`,
  :class:`~membrane.fragment.Prefix`,
  :class:`~membrane.fragment.Segment`,
  :class:`~membrane.fragment.Artifact`,
  :class:`~membrane.fragment.Trace`, and the discriminator
  :class:`~membrane.fragment_kind.FragmentKind`.
* Stable fragment identity: :class:`~membrane.identity.PayloadIdentity`
  (the durable ten-field fingerprint; this supersedes the removed
  ``membrane.types.Signature`` alias, which was deleted in 0.2.0).
* Serving plane: :class:`~membrane.node.Node`,
  :class:`~membrane.node.Origin`,
  :class:`~membrane.node.Replica`.
* Placement: :class:`~membrane.ring.Ring`,
  :class:`~membrane.shard.Shard`.
* Reconstruction: :class:`~membrane.reconstructor.Reconstructor`.
* Transfer: :class:`~membrane.transfer.TransferService`.
* Persistence: the :class:`~membrane.persistence.base.PersistenceBackend`
  protocol plus the
  :class:`~membrane.persistence.memory.Memory`,
  :class:`~membrane.persistence.redis.Redis`, and
  :class:`~membrane.persistence.cache.CachingPersistence`
  implementations.
* Compute: :class:`~membrane.compute.base.Backend` plus the
  :class:`~membrane.compute.cpu.CPU`,
  :class:`~membrane.compute.gpu.GPU`,
  :class:`~membrane.compute.transformers.Transformers`,
  :class:`~membrane.compute.openai.OpenAI`,
  :class:`~membrane.compute.anthropic.Anthropic`, and
  :class:`~membrane.compute.ollama.Ollama` backends.
* Transports: :class:`~membrane.transport.fastapi.FastAPIServer`.
* Composition: :class:`~membrane.server.Server`.
* Auth: the :class:`~membrane.auth.Authenticator` protocol
  (concrete ``APIKeyAuthenticator`` lives at
  ``membrane.auth.apikey``; ``TLSConfig`` at
  ``membrane.transport.tls``).
* Errors: the :class:`~membrane.errors.Error` hierarchy
  (``NetworkError``, ``SchemaError``, ``MigrationError``, etc.).
* Logging: :func:`~membrane.logging.configure_logging`.
* Wire format: the JSON envelope produced by
  :func:`~membrane.serialization.to_dict` / :func:`from_dict`
  (schema version 5; see ``docs/wire-format.md``).
* Prometheus metric names and label sets exposed by
  :mod:`membrane.metrics` and the FastAPI middleware.

## Per-module stability table

| Module path | Status | Notes |
|-------------|--------|-------|
| ``membrane.*`` (root exports) | **stable** (3.0+) | Mirrors ``membrane.__all__``. |
| ``membrane.network.cluster`` | **stable** (3.0+) | Cluster lifecycle; covered by chaos tests. |
| ``membrane.network.peer`` | **stable** (3.0+) | Peer-to-peer HTTP client. |
| ``membrane.network.gossip`` | **stable** (3.0+) | Gossip payload + merge. |
| ``membrane.network.membership`` | **stable** (3.0+) | Membership table + heartbeat loop. |
| ``membrane.network.config`` | **stable** (3.0+) | ``ClusterConfig`` + ``validate_config``. |
| ``membrane.compute.openai`` | **stable** (3.0+) | |
| ``membrane.compute.anthropic`` | **stable** (3.0+) | |
| ``membrane.compute.ollama`` | **stable** (3.0+) | |
| ``membrane.compute.transformers`` | **stable** (3.0+) | |
| ``membrane.compute.cpu`` / ``gpu`` | **stable** (3.0+) | |
| ``membrane.compute.remote`` | **stable** (3.0+) | Shared HTTP-LLM base class. |
| ``membrane.transport.fastapi`` | **stable** (3.0+) | |
| ``membrane.transport.ops`` | **experimental** | Per-op surface is being stabilised; helpers may move. |
| ``membrane.transport.grpc`` | **stable** (3.0+) | gRPC servicer + stubs. |
| ``membrane.transport.tls`` | **stable** (3.0+) | mTLS / TLS rotation helpers. |
| ``membrane.persistence.redis`` | **stable** (3.0+) | |
| ``membrane.persistence.memory`` | **stable** (3.0+) | |
| ``membrane.persistence.cache`` | **stable** (3.0+) | |
| ``membrane.security.encryption`` | **stable** (3.0+) | AES-256-GCM + ``DecryptError``. |
| ``membrane.security.url_allowlist`` | **stable** (3.0+) | SSRF guard. |
| ``membrane.security.tenant`` | **stable** (3.0+) | ``TenantAuthorizer``. |
| ``membrane.security.key_rotation`` | **stable** (3.0+) | |
| ``membrane.serialization`` | **stable** (3.0+) | Wire-format envelope + ``SCHEMA_VERSION``. |
| ``membrane.audit`` | **stable** (3.0+) | Append-only audit log. |
| ``membrane.transport.authz`` | **stable** (3.0+) | Per-route scope check. |
| ``membrane.secrets`` | **stable** (3.0+) | Backend-agnostic secrets vault. |
| ``membrane.wire.v3`` | **stable** (3.0+) | Generated gRPC stubs. |
| ``membrane.otel_tracer`` | **experimental** | Subject to minor-version renames. |
| ``membrane.disagg`` | **stable** (3.0+) | Prefill/decode disaggregation. |
| ``membrane.adapters`` | **stable** (3.0+) | Engine integration adapters. |
| ``membrane.quantization`` | **stable** (2.0+) | K/V tensor quantization. |
| ``membrane.metrics`` | **stable** (3.0+) | Prometheus collectors. |
| ``membrane.resilience`` | **stable** (3.0+) | Retry / circuit-breaker policies. |
| ``membrane.cli`` | **stable** (3.0+) | Typer + Rich CLI. |
| ``membrane.analytical`` | **stable** (3.0+) | Decision / policy classes. |
| ``membrane.canonical`` | **stable** (3.0+) | Canonical byte framing. |
| ``membrane.compat`` | **stable** (3.0+) | ``ModelCompatibilityFingerprint``. |
| ``membrane.errors`` | **stable** (3.0+) | Typed exception hierarchy. |

Internal modules (prefixed ``membrane._*``) are **not** covered
by this promise and may move without notice.

## Deprecation policy

* A deprecation warning is emitted in ``n.x`` for any breaking
  change planned for ``n+1.0``.
* Deprecations are documented in ``CHANGELOG.md`` with a
  removal target.
* Deprecations are removed no sooner than the minor version
  after their introduction.

## Wire format versioning

The on-wire format (HTTP and gRPC) carries a ``schema_version``
field managed by
:data:`membrane.serialization.SCHEMA_VERSION` (= 5). Bumping
the version is a breaking change and requires a major version
bump. See ``docs/wire-format.md`` for the full spec.

## References

* ``CHANGELOG.md`` — release history
* ``docs/release.md`` — release process
* ``docs/wire-format.md`` — on-wire / on-disk format
* ``docs/compat-matrix.md`` — runtime / engine compat
