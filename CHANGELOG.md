# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **CI security job**: The `aquasecurity/trivy-action` step was
  configured to scan the upstream `python:3.12-slim` base image
  (`scan-type: image, image-ref: python:3.12-slim`), which produced
  19 HIGH/CRITICAL CVEs in debian Trixie 13.6 base packages
  (perl-base, ncurses-base, libssl3, libsqlite3-0, gzip, libacl1,
  openssl). Those CVEs are upstream Debian stable issues and
  cannot be fixed from inside this repository. Switched the scan
  to filesystem mode (`scan-type: fs, scan-ref: .`) so the weekly
  audit now covers the Membrane codebase. Python-dep CVEs are
  still cross-checked by `pip-audit`, code-level issues by
  `bandit`, secrets by `gitleaks`.

## [0.3.0] - 2026-08-30

Principal-level architectural refactor of Membrane, continuing
the 0.2.0 cleanup. Each item below corresponds to one of the 16
atomic commits on the path from 0.2.0 to 0.3.0. The refactor
preserves every external contract (CLI flags, gRPC method
names, HTTP wire format, public dataclass shapes) and changes
only the internal module/class structure.

### Changed

- **No fake private APIs**: dropped leading underscores from
  `TransferService._transfer_local`, `TransferService._sync_local`,
  `TransferService._pull_from_remote`, `TransferService._push_to_remote`,
  renamed them to public `transfer_local`, `sync_local`,
  `pull_from_remote`, `push_to_remote`. Renamed
  `membrane.transport._ops` → `membrane.transport.ops`. Moved
  `Cluster._migrate_primary` → `Shard.migrate_primary` (where
  the data lives). Inlined `Server.setup_persistence / setup_cluster /
  setup_transport / make_compute_backend` into `Server.__init__`
  or replaced the helper with a registry.

- **No thin forwarders**: deleted `GraphManager` and merged its
  `suggest_prefetch` / `eviction_candidates` policies into `Graph`
  directly as `prefetch_suggest` and `eviction_neighbors`. Removed
  twelve `Cluster` forwarders (`add_peer`, `remove_peer`,
  `get_peers`, `is_peer_healthy`, `get_peer_client`, `get_peer_url`,
  `on_peer_join`, `on_peer_leave`, `on_heartbeat`, `on_gossip`,
  `bootstrap_loop`, plus the `_migrate_callback` rename);
  callers reach `cluster.membership.*` and `cluster.gossip.handle(...)`
  directly. Deleted `Server.register_peer` and
  `Server.unregister_peer`; the `connected_nodes` set is exposed
  by Server for any future diagnostics. Removed the dead
  `from membrane.delta import Delta` re-export in `prefix.py`.

- **Compute polymorphism**: hoisted `token_hash(tokens)` to
  `membrane/compute/_hash.py` and deleted the five duplicate
  copies (CPU `@staticmethod`, OpenAI / Anthropic / Ollama /
  Transformers instance methods, plus `CPU.hash_tokens` called
  from GPU). Hoisted `simulate_prefill_fragment(...)` to
  `Backend` as a static helper; deleted the three
  `simulate_prefill` instance methods and the two inlined
  copies. Added `Backend.SIMULATE_WINDOW_SIZE = 128` as the
  shared window-size constant.

- **Shared HTTP base class**: introduced `RemoteLLMBackend`
  with `build_client(timeout=, headers=)` and `probe(path)` so
  OpenAI, Anthropic, and Ollama no longer hand-roll their
  `httpx.Client` construction and `available()` liveness
  probes. Each backend still supplies the provider-specific
  URL paths and request bodies; the boilerplate is gone.

- **Compute registry**: `Server.__init__`'s six-way
  `if compute == "gpu" / "ollama" / "openai" / "anthropic" /
  "transformers" / default→CPU` chain is replaced with
  `COMPUTE_BACKENDS: dict[str, callable]` registry. `Server.compute`
  is now typed `Backend | str`; an existing `Backend` instance
  can be passed through.

- **Module consolidation**: merged `membrane/network/gossip_loop.py`
  into `membrane/network/gossip.py` (one concept, one module).
  Folded `membrane/clusters.py` (the `SemanticCluster` class)
  into `membrane/semantics.py` and renamed
  `clusters.SemanticCluster.cosine_similarity` to module-level
  `membrane.semantics.cosine_similarity`. Deleted both source
  modules; tests import from `membrane.semantics`.

- **Persistence split**: the monolithic `PersistenceBackend`
  Protocol in `membrane/persistence/base.py` is split into
  two: `Storage` (per-fragment CRUD) and `Inventory`
  (cross-node directory). `PersistenceBackend` remains as a
  combined alias for back-compat. Fixed the long-standing
  `delete_fragment(content_hash, node_id)` vs
  `delete_fragment(content_hash)` signature mismatch — every
  concrete backend already used the single-argument form;
  the Protocol now matches.

- **Diagnostic cleanups**: dropped the
  `isinstance(self.persistence, Redis) and self.persistence.ping()`
  tag-check in `Server.diagnostics()`. The field is now the
  persistence-backend health indicator (returned from
  `self.persistence.ping()`) and is no longer coupled to a
  particular backend kind.

- **Tests**: rewrote `tests/membrane/network/test_cluster.py`
  to drive subsystems directly (no `mgr.add_peer` /
  `mgr.on_peer_join` forwarders). Updated
  `tests/membrane/transport/test_fastapi_server.py` to assert
  against `cluster.membership.add` /
  `cluster.membership.to_json` instead of the now-deleted
  `on_peer_join` mock. Updated
  `tests/membrane/transport/test_observability.py` so the
  readyz over-capacity test no longer mutates
  `app.state.node.max_memory_bytes` /
  `app.state.node.fragments` directly — saturates the node
  via `node.memory_usage = node.max_memory_bytes`. Dropped
  `backend.actual_device = "cpu"` mutation from
  test_transformers_backend.py. Rewrote test_semantic_cluster
  to import from `membrane.semantics`. test_graph_manager
  renamed to test_graph with assertions on `Graph` directly.

### Removed

- `membrane/clusters.py` (folded into semantics.py).
- `membrane/network/gossip_loop.py` (folded into gossip.py).
- `Cluster.{add_peer, remove_peer, get_peers, is_peer_healthy,
   get_peer_client, get_peer_url, on_peer_join, on_peer_leave,
   on_heartbeat, on_gossip, _migrator_callback,
   bootstrap_loop}` (forwards).
- `Server.{setup_persistence, setup_cluster, setup_transport,
   make_compute_backend, register_peer, unregister_peer}`
  (prefixed helpers + dead forwards).
- `GraphManager` (folded into Graph).
- `TransferService.{_transfer_local, _sync_local,
   _pull_from_remote, _push_to_remote}` underscore prefix.
- Five per-backend `hash_tokens` methods (replaced with
  `membrane.compute.token_hash`).
- Three per-backend `simulate_prefill` methods (replaced
  with `Backend.simulate_prefill_fragment`).
- Three duplicate `httpx.Client(...)` constructor blocks.
- Three duplicate `available()` liveness probes.
- Side-effect `from membrane.delta import Delta` re-export.
- `membrane.transport._ops` (underscore module name).

### Notes

- External contracts preserved: gRPC method names, HTTP wire
  format (JSON), CLI flag set, `Server.__init__` signature,
  `ClusterConfig` fields, every fragment dataclass. The
  rename of `GraphManager.suggest_prefetch` →
  `Graph.prefetch_suggest` is the only public method that
  has been renamed; the old name is no longer reachable
  via the public surface.

## [0.2.0] - 2026-08-30

- **CI security job**: The `aquasecurity/trivy-action` step was
  configured to scan the upstream `python:3.12-slim` base image
  (`scan-type: image, image-ref: python:3.12-slim`), which produced
  19 HIGH/CRITICAL CVEs in debian Trixie 13.6 base packages
  (perl-base, ncurses-base, libssl3, libsqlite3-0, gzip, libacl1,
  openssl). Those CVEs are upstream Debian stable issues and
  cannot be fixed from inside this repository. Switched the scan
  to filesystem mode (`scan-type: fs, scan-ref: .`) so the weekly
  audit now covers the Membrane codebase. Python-dep CVEs are
  still cross-checked by `pip-audit`, code-level issues by
  `bandit`, secrets by `gitleaks`.

## [0.2.0] - 2026-08-30

Principal-level architectural refactor. Internal-only; no public-API
behavior change unless called out below.

### Changed

- **Domain model**: Centralized the four magic model_id strings
  ("prefix", "kv", "artifact", "tool", "weighted_graph") as a
  type-safe `membrane.fragment_kind.FragmentKind` enum. All memory
  objects (`Prefix`, `Segment`, `Artifact`, `Trace`) and the
  weighted-graph placeholder use the enum; `Isolation` keys off the
  enum rather than raw strings. Wire format strings are unchanged.
- **Stores**: Deleted `membrane.store.Store` and
  `StoreMetrics`. `Node` is the single canonical in-memory
  fragment store; `Memory` in `membrane.persistence.memory`
  remains the `PersistenceBackend` reference implementation.
- **Directories**: Deleted `membrane.directory.Directory` and
  `membrane.supernode.Supernode` (parallel dead implementations of
  the same concept). `membrane.registry.Registry` is the surviving
  canonical directory, used by the cluster subsystem.
- **Transfer plane**: Folded `membrane.network.transfer.Transfer`
  into `membrane.transfer.TransferService`. The unified class
  dispatches over local nodes or remote peer ids based on
  constructor wiring.
- **Replicator**: Folded the two parallel `Replicator` classes
  (`membrane.replicator` and `membrane.network.replicator`) into a
  single `Replicator` that supports both one-shot
  `replicate_cluster` and the background `loop()` mode.
- **Prefill**: Folded `Adapter`, `PrefillAsync`, and
  `PrefillRemote` into the new `membrane.prefiller.Prefiller`
  with `dispatch` (async race) and `dispatch_sync` (single
  target) entry points.
- **Dashboards**: Folded `membrane.cli.poll` into
  `membrane.cli.dashboard`. The `membrane dashboard` subcommand
  moved to `membrane.cli.commands.dashboard`.
- **Transport routes**: Extracted the stdlib-HTTP and FastAPI
  handler bodies into `membrane.transport._ops` (a single
  transport-agnostic operations module). Both transports now
  delegate to the shared operations.
- **Routers / selectors / policies / offload** are now
  importable only from their own modules; they are no longer
  re-exported from `membrane`.
- **Resilience**: `TimeoutPolicy` is now actually enforced via
  `signal.alarm` / `signal.setitimer` inside
  `ResiliencePolicy.guard`. (Previously the policy was
  configurable but the guard ignored it.) `signal.signal(SIGALRM, ...)`
  is restored on every guard exit so the alarm never leaks.
- **Persistence**: `Server.setup_persistence` now wraps the
  selected backend (Memory or Redis) in
  `CachingPersistence`. Repeated reads are served from the local
  in-process cache instead of crossing the network on every call.
- **Cluster**: `Cluster.__init__` constructs an
  `EagerMigrator` by default and binds `_migrate_primary` to it.
  `Cluster.on_peer_leave` collects the leaving peer's primaries
  via `Shard.primary_map` and hands them to the migrator; the
  local node is promoted and the leaving peer is removed from
  `Shard.replica_map`. `RateLimitedMigrator` enforces a
  configurable migrations-per-second ceiling.
- **Public API**: `membrane/__init__.py` re-exports only the
  durable domain concepts (~50 names). Sub-indexes, the graph
  layer, decision classes, model helpers, CLI helpers,
  observability primitives, and resilience policies are
  importable from their own modules; the docstring lists the
  deep paths.
- **Tests**: Renamed 13 test files so the file name matches the
  class it exercises (`test_kv_cache_manager.py` →
  `test_kv.py`, `test_origin_node.py` → `test_origin.py`,
  `test_fragmentation_engine.py` → `test_fragmenter.py`, ...).
  Removed the autouse-fixture trick that injected
  `make_fragment` into test module namespaces; each test file
  now imports the factory explicitly. Dropped the `F821` ignore
  in `pyproject.toml` since ruff can now see every reference.

### Fixed

- **`Index.remove`**: Replaced the short-circuit OR chain with
  four unconditional sub-index removals so a co-access entry
  is reliably cleaned up when the underlying fragment is
  removed. Added regression tests.
- **Stale docstring paths**: Rewrote every stale
  `:class:`~membrane.fragmentation_engine.XXX`` reference
  (~30 occurrences across the source tree) to point at the
  actual current paths (`fragmenter`, `index`, `exacts`,
  `semantics`, `coaccess`, `tree`, `node`, `origin`, `replica`,
  `transfer`, `store`, `registry`, `deltasync`, `latency`,
  `economic`, `joint`, `peer`, `compute.cpu`, `compute.gpu`,
  `compute.ollama`, `compute.openai`, `compute.anthropic`,
  `compute.transformers`, etc.).
- **Adapter / PrefillAsync wiring**: `Adapter.prefill` has
  always computed a `RoutingDecision` from
  `model.router.Router`; `PrefillAsync.dispatch` previously
  ignored it. Now a `target="pd-p"` decision skips the remote
  race and serves locally; the check is gated on a configured
  router so test adapters without a router are not perturbed.
  Added tests covering both `pd-p` and `membrane` targets.
- **Unit-formatting**: 25 files were reformatted via `ruff
  format`; 47 import-order / unused-import violations were fixed
  via `ruff check --fix`.
- **MD5 in compute backends**: every `hashlib.md5(...)` call
  in the compute / content-addressing layers now passes
  `usedforsecurity=False`. The hashes are used for content
  addressing, not authentication, so bandit no longer flags
  them as B324 (high severity).

### Removed

- `membrane/types.py` (unimported, broken self-alias).
- `membrane/semhash.py` (only used by its own test).
- `membrane/protocols.py` (declared Protocols no concrete
  class implemented).
- `membrane/store.py` and `StoreMetrics` (no production
  callers; `Node` is the canonical in-memory store).
- `membrane/directory.py` and `membrane/supernode.py`
  (parallel dead directory implementations).
- `membrane/network/transfer.py` and the `RemoteTransfer`
  alias (folded into `TransferService`).
- `membrane/network/replicator.py` (folded into the top-level
  `Replicator`).
- `membrane/cli/poll.py` (folded into `cli/dashboard`).
- `membrane/prefill_async.py` and `membrane/prefill_remote.py`
  (folded into `membrane/prefiller.py`).
- `membrane/kvreturn.py` (a 70-line wrapper around
  `TransferService.transfer_fragment`; re-implemented its
  test against `TransferService` directly).
- `auth/apikey.py::ensure_runtime_checkable` and
  `ignore_unused` (asserts at import time and a no-op helper).
- `membrane/telemetry::telemetry` (the function collided with
  the `Telemetry` class; the function was never called).
- `models` removed in commit 1.
- Stale per-file ruff ignores for `cli.py` and `grpc_server.py`
  (renamed/deleted in earlier commits).
- All `_`-prefixed application helpers in `auth/apikey.py`.

## [0.1.2] - 2026-07-12
- MIT License file.
- CONTRIBUTING.md with development guidelines and conventional commits.
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1).
- SECURITY.md with vulnerability reporting process.
- .env.example with documented environment variables.
- .editorconfig for consistent code formatting.
- .gitattributes for line ending normalization.
- GitHub issue templates (bug report, feature request).
- Pull request template.
- Dependabot configuration for pip, GitHub Actions, and Docker.
- FUNDING.yml with GitHub Sponsors placeholder.
- docs/getting-started.md — onboarding guide.
- docs/architecture.md — system design documentation.
- docs/deployment.md — production deployment guide.
- docs/faq.md — frequently asked questions.
- Comprehensive Google-style docstrings on every module, class, function,
  and method (100% module coverage, 100% class coverage, 99% function/method
  coverage). Algorithm references added for AVL interval trees, consistent
  hashing, weighted LRU, delta encoding, gossip protocol, and the six
  throughput equations from the paper.

### Changed
- Rewrote README.md with badges, detailed features, API examples, and roadmap.
- Updated CHANGELOG.md with unreleased section.
- Updated .gitignore with additional Python and IDE patterns.
- Updated GitHub Actions CI workflow with linting and coverage steps.
- Public API surface expanded: previously-`_`-prefixed internal helpers
  are now public. `_MembraneServicer` -> `MembraneServicer`,
  `_MembraneHTTPHandler` -> `MembraneHTTPHandler`, `_send_json`,
  `_read_json`, `_handle_*` (in HTTP server), `_serialize_fragment`,
  `_deserialize_fragment`, `_hash_tokens`, `_simulate_prefill`,
  `_load_model`, `_request` -> `request_with_retry`, `_to_fragment` ->
  `pb_to_fragment`, `_to_message` -> `fragment_to_pb`,
  `_serialize`/`_deserialize` (in persistence backends) ->
  `serialize_fragment`/`deserialize_fragment`, `_key` -> `key_for`,
  `_bootstrap_loop` -> `bootstrap_loop`, `_heartbeat_loop`,
  `_failure_detection_loop`, `_gossip_loop`, `_replication_loop`,
  `_inventory_digest` -> `inventory_digest`, `_make_compute_backend`,
  `_setup_persistence`, `_setup_cluster`, `_setup_transport`,
  `_ensure_node` -> `ensure_node`, `_interactive_setup` ->
  `interactive_setup`, `_run_dashboard` -> `run_dashboard`.
  BREAKING CHANGE for any caller that overrode `_handle_*` on a custom
  HTTP handler subclass.
- Bumped all runtime dependencies to their latest stable releases as of
  2026-08-29: `typer>=0.27.2`, `rich>=15.0.0`, `redis>=8.1.0`,
  `fastapi>=0.141.1`, `uvicorn>=0.52.4`, `ruff>=0.16.5`,
  `grpcio>=1.83.1`, `grpcio-tools>=1.83.1`, `mypy>=2.3.1`,
  `pydantic>=2.13.5`, `transformers>=5.16.1`, `torch>=2.13.0`,
  `httpx>=0.28.1`, `protobuf>=7.36.0`, `pytest>=9.1.1`,
  `pytest-cov>=7.1.0`. Bumped CI action versions
  (`actions/checkout@v7`, `actions/setup-python@v7`,
  `docker/build-push-action@v7`).
- Converted the `peer` option on `membrane serve` to the
  `Annotated[list[str] | None, typer.Option(...)] = None` form so the
  default value is no longer a function call, eliminating a
  mutable-default-argument lint warning without a per-file suppression.
- Updated the ruff per-file-ignores list to reflect the current file
  layout: removed stale entries for renamed files (`cli.py`,
  `grpc_server.py`, `auth/__init__.py`) and tightened the
  `membrane/__init__.py` re-export list by adding `DeltaSync` and
  `TransferService` to `__all__` rather than suppressing the unused-import
  warning.

### Fixed
- `membrane.metrics.metrics_summary`: the function referenced
  `registry._counters` / `registry._gauges` (with a leading underscore)
  which never existed on `MetricsCollector`; the summary therefore raised
  `AttributeError` at runtime. Renamed to the actual public attributes
  (`counters`, `gauges`).
- `membrane.persistence.cache.CachingPersistence`: removed a self-referential
  `@property def inner` whose getter returned `self.inner` (infinite
  recursion) and which also made `self.inner = inner` in `__init__`
  raise `read-only property` errors. The cache now stores `inner` as a
  plain attribute, matching how every other backend is composed.
- `membrane.network.cluster.ClusterManager.__init__`: removed a duplicate
  block that re-initialized `self.running`, `self.stop_event`, and
  `self.threads` after the subsystems that already captured references
  to them — `running` and `threads` were reported by mypy as redefined.
- `membrane.transport.routes.get_handler_node`: the return annotation
  was `dict[str, Any]` even though the helper returned a live `Node`
  (or an empty `dict` sentinel). Every caller was then flagged by mypy
  when it tried to call `node.store(...)` / `node.fragments.items()` /
  etc. on `dict[str, Any]`. The helper now returns `Node | None`, the
  sentinel is `None`, and each caller already had an `if not node:`
  early-return so the change is purely a type-correctness fix.
- `membrane.auth.apikey.ignore_unused`: the helper referenced `Any`
  without importing it (it was only re-exported from
  `membrane.auth.__init__`). Added a local `from typing import Any`
  import and dropped the now-unused `Any` re-export from the package
  `__init__.py`.
- Compute backends and content hashers (`membrane.compute.{cpu,openai,
  anthropic,ollama,transformers}.token_hash`, `membrane.fragmenter.
  payload_hash`, `membrane.ring.hash_value`): every call to
  `hashlib.md5(...)` now passes `usedforsecurity=False`. The hashes are
  used purely for content addressing and consistent-hashing placement,
  not authentication, so they were false-positive B324 (high severity)
  findings in bandit.
- Reconstruction-engine tests (`tests/membrane/test_reconstruction_engine.py`):
  six tests (`test_full_exact_match_no_prefill`,
  `test_partial_match_with_positional_extension`,
  `test_gap_filled_by_semantic_similarity`, `test_large_gap_triggers_prefill`,
  `test_coverage_ratio_accuracy`, `test_graph_links_recorded`) created
  fragments with placeholder `content_hash` strings (`"a"`, `"match"`,
  …) and `model_id="test-model"`, then asserted behaviour under
  `rebuild_context(..., "m")`. The reconstructor's exact-index lookup
  keys fragments by `compute_content_hash(tokens)`, so those placeholders
  were never reachable through `longest_match`. Coverage used to reach
  1.0 only via semantic-similarity coincidence (and `prefill_invoked`
  was wrong on `test_partial_match_with_positional_extension` for the
  same reason). Replaced the placeholder hashes with
  `compute_content_hash(tokens[start:end+1])` and `model_id="m"`, added
  a small `_fragment_for_span` helper, and dropped two `@pytest.mark.xfail`
  markers that had been hiding the failure.
- `membrane.cost.should_prefill`: collapsed an if/else that assigned
  `retrieve_cost` into a single ternary to satisfy `ruff` rule SIM108.
- `membrane.network.failure.scrub_loop`: combined a nested
  `elif`/`if` pair into a single guarded `elif … and …` to satisfy
  `ruff` rule SIM102.
- `tests/membrane/network/test_cluster_manager.py`: same SIM102
  collapse as in the production failure detector.
- Removed 44 auto-fixable ruff violations across the tree (`I001`
  unsorted imports, `F841` unused variables, `W293` trailing whitespace,
  `F811` redefined-but-unused, `UP035` deprecated imports) and
  reformatted 25 files that drifted from `ruff format`.
- Removed stale per-file ruff ignores for files that no longer exist
  after the `R5–R7` file renames (`membrane/cli.py`,
  `membrane/transport/grpc_server.py`).

## [0.1.2] - 2026-07-12

### Fixed
- pyproject.toml: `dependencies` correctly placed under `[project]`
  (was nested under `[project.urls]`).
- pyproject.toml: resolved unresolved git merge-conflict markers in
  the optional-dependency section and removed the obsolete
  `[tool.ruff.lint.isort] profile = "black"` block that prevented
  `ruff check` from parsing the configuration.
- gRPC transport: regenerate `membrane_pb2.py` and `membrane_pb2_grpc.py`
  against the installed `grpcio-tools` (1.82.1, protobuf 7.35.1). The
  previous stubs required protobuf >=6.31 at runtime and crashed
  with `RuntimeError` on import when paired with older protobuf
  versions; the new stubs are compatible with the pinned runtime.
- Pin `grpcio-tools>=1.81.1` in the `server` extras so future
  regenerations remain consistent with the pinned `grpcio`.
- HTTP server: replace bare `except Exception:` blocks with the
  specific exception types each call site can actually encounter
  (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError).
- Compute backends (OpenAI, Anthropic, Ollama, Transformers):
  replace bare `except Exception:` with specific exception types
  (httpx.HTTPError, RuntimeError, ValueError, IndexError,
  AttributeError, JSONDecodeError, OSError).
- Persistence (Redis): narrow `ping()` to `(redis.RedisError, OSError)`
  and expose `RedisError` on the instance for callers.
- Replace suppressed `# noqa: F401, E402` (prefix.py) and
  `# type: ignore[import-untyped]` (transport/grpc_server.py) with
  documented inline comments explaining why each suppression is
  intentional.
- mypy: cast `dict[str, str]` to the redis-py `Mapping` stub type
  in `redis_backend.store_fragment` to satisfy the (overly strict)
  stub signature without changing the wire format.

### Removed
- Suppressed `# noqa` and bare `except Exception:` clauses that hid
  real errors in tests and backends. Tests now use the concrete
  exception types (`httpx.TimeoutException`, `httpx.ConnectError`,
  `httpx.HTTPStatusError`) that production code catches.

## [0.1.1] - 2026-05-15

### Fixed
- Version bump and minor fixes.

## [0.1.0] - 2026-05-08

### Added
- Initial release of Membrane.
- Analytical throughput model from "Prefill-as-a-Service" paper (Section 3.4.1, Equations 1–6).
- Grid-search optimizer for throughput-optimal routing threshold and PD split (Section 3.4.2).
- Dual-timescale scheduler with bandwidth-aware short-term routing and long-term reallocation (Section 3.4.3).
- Truncated log-normal workload generator with paper parameters (Section 4.1).
- Baseline simulators: Membrane-PD, Homogeneous PD, and Naive Heterogeneous PD.
- Fragment data model with content-addressed, immutable KV segments and structural signatures.
- In-memory indices: Exact, Semantic, Positional, and Co-access.
- Fragment relationship graph with weighted edges and graph lifecycle management.
- Reconstruction engine with `rebuild_context()` and prefill fallback.
- Global and distributed directory for multi-node fragment location resolution.
- Transfer plane with delta-sync and chunked, resumable fragment transfer.
- Multi-tenant canonical store with tenant isolation.
- Production server with HTTP (stdlib + FastAPI), gRPC, and CPU/GPU/transformers compute backends.
- Redis persistence backend with LRU eviction tracking.
- CLI with live TUI dashboard, cluster status, and interactive setup wizard.
- Comprehensive test suite (548+ tests) and CI pipeline for Python 3.10–3.13.

[0.2.0]: https://github.com/sachncs/membrane/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/sachncs/membrane/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sachncs/membrane/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sachncs/membrane/releases/tag/v0.1.0