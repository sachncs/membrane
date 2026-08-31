"""Commit the pytest-benchmark baseline for the v3.0.1 release.

The 3.6.7 commit shipped the bench surface; this commit pins
a machine-local baseline under ``.benchmarks/3.0.1.json`` so
subsequent 3.0.x patch releases can compare against the v3.0.1
timings without re-running the bench on the developer's box.

Regenerate with::

    pytest tests/bench/test_phase37_bench.py \\
        --benchmark-only \\
        --benchmark-json=.benchmarks/3.0.1.json

pytest-benchmark stores its own file-per-commit cache in
``.benchmarks/`` on the developer's box; this commit only
captures a *human-readable* baseline for review.
"""

from __future__ import annotations

import pytest


class TestBenchmarkBaseline:
    """Sanity-check the bench surface compiles + runs."""

    def test_baseline_module_is_importable(self):
        # The bench surface is opt-in; the baseline file under
        # .benchmarks/3.0.1.json is committed for human review.
        # The test only asserts the module imports cleanly so the
        # bench surface can run via ``pytest tests/bench/`` on
        # developer / CI boxes that have pytest-benchmark
        # installed.
        import importlib
        importlib.import_module("tests.bench.test_phase37_bench")
