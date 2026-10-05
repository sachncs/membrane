# Contributing to Membrane

Thank you for your interest in contributing to Membrane! This document provides guidelines and instructions to help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/membrane.git
   cd membrane
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/sachncs/membrane.git
   ```
4. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feat/my-new-feature master
   ```

## Development Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs Python 3.14, the only
  supported version (pinned in `.python-version`).
- (Optional) Redis for persistence backend testing

### Install dependencies

```bash
uv sync --frozen --extra dev     # exact versions from uv.lock
source .venv/bin/activate        # Linux/macOS
```

`--frozen` installs exactly what `uv.lock` records, as CI does. When you
change dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`.

### Optional extras

```bash
uv sync --frozen --extra dev --extra transfer    # numpy, lz4 (zstd is stdlib)
uv sync --frozen --extra dev --extra gpu         # PyTorch CUDA
uv sync --frozen --extra dev --extra local-llm   # HuggingFace Transformers
```

## Branch Naming

Use descriptive branch names with a type prefix:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

Example: `feat/add-kubernetes-operator`

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (dependencies, CI, etc.) |
| `perf` | Performance improvements |

### Examples

```
feat(reconstruction): add fallback to prefill on cache miss
fix(indices): correct semantic index threshold calculation
docs: update deployment guide with Kubernetes instructions
test(fragment): add edge case tests for empty fragments
chore: update pytest to 8.x
```

## Pull Request Process

1. **Ensure your branch is up to date** with `master`:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run the checks CI runs** before submitting:
   ```bash
   pytest tests/
   ruff check membrane tests scripts examples
   ruff format --check membrane tests scripts examples
   mypy membrane
   python tools/check_naming.py      # naming, __all__, no direct output
   python tools/check_docstrings.py  # complete Google-style docstrings
   ```
   CI's `CI passed` check must be green before merge.

3. **Push your branch** and open a Pull Request against `master`.

4. **Fill out the PR template** completely, including:
   - Summary of changes
   - Related issue (if any)
   - Testing done
   - Checklist confirmation

5. **Request review** from a maintainer.

6. **Address review feedback** promptly. Push additional commits to your branch as needed.

7. **Merge** will be handled by a maintainer once approved.

## Coding Standards

### General

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use type hints for all function signatures. Annotations are deferred
  (PEP 649), so do not add `from __future__ import annotations`.
- Every module, class, and function (private ones included) has a
  Google-style docstring with `Args:`, `Returns:`, `Yields:`, and
  `Raises:` sections as applicable; `tools/check_docstrings.py` and
  ruff's `D` rules enforce it.
- Mark overriding methods with `@typing.override`; mypy enforces it.
- Write Python 3.14: `type` aliases, `match`, `compression.zstd`,
  `uuid.uuid7()`, template strings for structured logs.

### Output

- Never `print`. Diagnostics use `logging.getLogger(__name__)`. CLI
  commands report through `membrane.cli.output`: `result()` and
  `result_json()` write to stdout for piping, `info()` and `error()`
  write to stderr. ruff `T20` and `tools/check_naming.py` reject `print`,
  `console.print`, `typer.echo`, and `sys.stdout.write`.

### Formatting and linting

- Format with `ruff format` and lint with `ruff check` (configured in
  `pyproject.toml`; import sorting is part of the lint).

### Type Checking

- All code must pass `mypy` with the project configuration in `pyproject.toml`.

### Naming

- Use `snake_case` for functions, methods, and variables.
- Use `PascalCase` for classes and exceptions.
- Use `UPPER_SNAKE_CASE` for constants.
- No single-underscore names. A name is either public, or private to
  its class with a double underscore (`__name`, name-mangled). Module
  APIs are declared in `__all__`, which every module defines.
  `tools/check_naming.py` enforces this.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=membrane --cov-report=term-missing

# Run one test file
pytest tests/membrane/test_fragment.py -v

# Redis integration tests (skip without a server on localhost:6379)
docker run -d --rm -p 6379:6379 redis:7-alpine
pytest tests/membrane/persistence

# Run type checking
mypy membrane
```

## Documentation

- Update documentation when changing public APIs.
- Add docstrings to new public functions and classes.
- Update `CHANGELOG.md` under the `[Unreleased]` section.
- Keep `README.md` and `docs/` current with new features or setup
  changes. `docs/` is published to the website; `cd site && npm ci &&
  npm run dev` previews it.

## Questions?

Open a [GitHub Discussion](https://github.com/sachncs/membrane/discussions) if you have questions or need help getting started.
