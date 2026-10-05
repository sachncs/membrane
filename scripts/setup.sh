#!/usr/bin/env bash
# Set up a development environment from the lockfile and run the checks CI runs.
set -euo pipefail

echo "=== Membrane Setup ==="

if ! command -v uv >/dev/null; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# Python 3.14 (from .python-version) and the exact versions in uv.lock.
uv sync --frozen --extra dev

echo "Verifying package import..."
uv run python -c "import logging, membrane; logging.basicConfig(level=logging.INFO, format='%(message)s'); logging.info('Package OK: membrane %s on Python 3.14', membrane.__version__)"

echo "Running checks..."
uv run ruff check membrane tests scripts examples
uv run mypy membrane
uv run python tools/check_naming.py
uv run python tools/check_docstrings.py
uv run pytest tests/ -q --tb=short

echo "=== Setup Complete ==="
