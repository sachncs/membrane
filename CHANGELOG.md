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

### Changed
- Rewrote README.md with badges, detailed features, API examples, and roadmap.
- Updated CHANGELOG.md with unreleased section.
- Updated .gitignore with additional Python and IDE patterns.
- Updated GitHub Actions CI workflow with linting and coverage steps.

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

[Unreleased]: https://github.com/sachn-cs/membrane/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sachn-cs/membrane/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sachn-cs/membrane/releases/tag/v0.1.0
