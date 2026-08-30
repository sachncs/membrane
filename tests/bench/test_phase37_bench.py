"""pytest-benchmark suite (Phase 3.6.7).

A minimal benchmark surface that operators can run with
``pytest --benchmark-only`` once the optional
``pytest-benchmark`` plugin is installed. The benchmarks
exercise the v3.0.0 hot path: enqueue a Fragment on the
Node, hit it once, retrieve once. Output is a JSON file
under ``.benchmarks/`` that the CI smoke job compares
against the committed baseline.

This module is skipped when :mod:`pytest_benchmark` is not
installed; the smoke fallback lives in
:mod:`tests.bench.test_phase37_smoke`.
"""

from __future__ import annotations

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity

try:
    import pytest_benchmark

    _BENCHMARK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BENCHMARK_AVAILABLE = False

import pytest


def _fragment(payload_size: int) -> Fragment:
    """Build a fragment of ``payload_size`` bytes."""
    ident = PayloadIdentity(
        payload_hash=f"hash-{payload_size}".ljust(64, "0")[:64],
        model_id="bench",
        model_revision="",
        tokenizer_name="bench",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, payload_size),
        dtype="float16",
        shape=(1, 1, 1, 8, 64),
    )
    return Fragment(
        identity=ident,
        payload_ref=None,
        payload_size=payload_size,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
        tenant_id="public",
    )


@pytest.mark.skipif(
    not _BENCHMARK_AVAILABLE, reason="pytest-benchmark plugin not installed"
)
def test_store_and_retrieve_small(benchmark):
    """Benchmark a 64-byte store / retrieve cycle."""
    from membrane.node import Node

    def cycle() -> None:
        node = Node(node_id="bench", max_memory_bytes=1_000_000)
        frag = _fragment(64)
        node.store(frag, is_primary=True)
        node.retrieve(frag.identity.payload_hash)
        node.access_times[frag.identity.payload_hash] = 1.0

    benchmark(cycle)


@pytest.mark.skipif(
    not _BENCHMARK_AVAILABLE, reason="pytest-benchmark plugin not installed"
)
def test_hash_chain_append(benchmark):
    """Benchmark 100 audit-log appends."""
    from membrane.audit import AuditLog

    log = AuditLog()

    def cycle() -> None:
        for i in range(100):
            log.record(actor="bench", action="store", payload={"i": i})

    benchmark(cycle)


@pytest.mark.skipif(
    not _BENCHMARK_AVAILABLE, reason="pytest-benchmark plugin not installed"
)
def test_chunk_manifest_round_trip(benchmark):
    """Benchmark chunk-manifest build + verify on a 16 KiB payload."""
    from membrane.wire.v3 import ChunkManifest

    payload = b"x" * (16 * 1024)

    def cycle() -> None:
        manifest = ChunkManifest.from_payload(
            payload=payload,
            content_hash="0" * 64,
            chunk_size=4096,
        )
        manifest.verify_chunk(0, payload[:4096])

    benchmark(cycle)

