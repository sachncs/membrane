# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/sachncs/membrane/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/sachncs/membrane/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sachncs/membrane/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sachncs/membrane/releases/tag/v0.1.0