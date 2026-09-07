"""Lightweight benchmark smoke that runs without pytest-benchmark.

The CI job falls back to this surface when the
``pytest-benchmark`` plugin is unavailable so a smoke
result is always present in the test output.
"""

from __future__ import annotations

import time

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


def _fragment(payload_size: int) -> Fragment:
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


def test_store_and_retrieve_smoke():
    """Smoke benchmark: store + retrieve 100 fragments and report the wall-clock.

    The test asserts a soft ceiling of 5 seconds; on a slow CI
    runner the assertion is loose, the printed number is the
    artifact operators eyeball in the logs.
    """
    from membrane.node import Node

    node = Node(node_id="bench", max_memory_bytes=10_000_000)
    frags = [_fragment(64 + i) for i in range(100)]
    start = time.perf_counter()
    for frag in frags:
        node.store(frag, is_primary=True)
        node.retrieve(frag.identity.payload_hash)
    elapsed = time.perf_counter() - start
    print(f"\n[bench] 100x store+retrieve: {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
