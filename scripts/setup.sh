#!/usr/bin/env bash
set -euo pipefail

echo "=== Membrane Setup ==="

# Verify Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-cov mypy

# Verify package imports
echo "Verifying package import..."
python -c "import membrane; print(f'Package OK: {len(membrane.__all__)} exports')"

# Run type check
echo "Running mypy..."
python -m mypy membrane/ --ignore-missing-imports

# Run tests
echo "Running tests..."
python -m pytest tests/ -q --tb=short

echo "=== Setup Complete ==="
