#!/usr/bin/env bash
set -euo pipefail

echo "=== Membrane Cleanup ==="

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
find . -type f -name '*.py.class' -delete 2>/dev/null || true

# Remove coverage data
rm -f .coverage
rm -rf .pytest_cache/

# Remove build artifacts
rm -rf build/ dist/ *.egg-info/

echo "=== Cleanup Complete ==="
