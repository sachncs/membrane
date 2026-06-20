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
   git remote add upstream https://github.com/sachn-cs/membrane.git
   ```
4. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feat/my-new-feature master
   ```

## Development Setup

### Prerequisites

- Python 3.10 or later
- pip
- (Optional) Redis for persistence backend testing

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e ".[dev]"
```

### Optional extras

```bash
# Server dependencies (FastAPI, gRPC, Redis)
pip install -e ".[server]"

# GPU backend (PyTorch CUDA)
pip install -e ".[gpu]"

# Local LLM backend (HuggingFace Transformers)
pip install -e ".[local-llm]"
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

2. **Run the full test suite** before submitting:
   ```bash
   pytest tests/ -v
   python -m mypy membrane/
   ```

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
- Use type hints for all function signatures.
- Write docstrings for public APIs using Google-style format.

### Formatting

- Use [Black](https://github.com/psf/black) for code formatting.
- Use [isort](https://pycqa.github.io/isort/) with the `black` profile for import sorting.

### Type Checking

- All code must pass `mypy` with the project configuration in `pyproject.toml`.

### Naming

- Use `snake_case` for functions, methods, and variables.
- Use `PascalCase` for classes and exceptions.
- Use `UPPER_SNAKE_CASE` for constants.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=membrane --cov-report=term-missing

# Run specific test file
pytest tests/test_fragment.py -v

# Run type checking
python -m mypy membrane/
```

## Documentation

- Update documentation when changing public APIs.
- Add docstrings to new public functions and classes.
- Update `CHANGELOG.md` under the `[Unreleased]` section.
- Keep `README.md` current with new features or setup changes.

## Questions?

Open a [GitHub Discussion](https://github.com/sachn-cs/membrane/discussions) if you have questions or need help getting started.
