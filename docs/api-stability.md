# API Stability Promise

Membrane's public API follows [Semantic Versioning](https://semver.org/)
with the additional guarantees below.

## Stable API surface

The following is **stable** from `v1.0.0` onward:

* The `membrane.types` façade: `Fragment`, `Node`, `Signature`, etc.
* `membrane.metrics.MetricsCollector` and typed collectors.
* `membrane.resilience.ResiliencePolicy` and its strategy classes.
* `membrane.auth.Authenticator`, `APIKeyAuthenticator`, `TLSConfig`.
* `membrane.errors.*` exception hierarchy.
* HTTP and gRPC endpoint contracts (URL, method, request/response schema).
* Prometheus metric names and labels.

## Deprecation policy

* A deprecation warning is emitted in `n.x` for any breaking change
  planned for `n+1.0`.
* Deprecations are documented in `CHANGELOG.md` with a removal target.
* Deprecations are removed no sooner than the minor version after their
  introduction.

## Wire format versioning

The on-wire format (HTTP and gRPC) carries a `schema_version` field
managed by `membrane.serialization.SCHEMA_VERSION`. Bumping the version
is a breaking change and requires a major version bump.

## What is NOT stable

* Module paths other than those listed above may change between minor
  versions. Internal modules (e.g., `membrane.network.cluster`) follow
  the package layout but are not guaranteed stable.
* Class names that exist primarily for grouping (e.g., transport
  wrappers) may be renamed in minor versions; they are not part of the
  stable contract.

## References

* `CHANGELOG.md` — release history
* `docs/release.md` — release process
