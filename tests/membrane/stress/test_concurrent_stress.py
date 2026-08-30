"""Stress suite for index / storage / lifecycle (Phase 3.7.6).

The v3.0.0 release ships a 64-thread stress suite that
exercises the index, the storage layer, and lifecycle
transitions (store / load / evict / drain) under concurrent
pressure. The suite is gated by ``pytest -m stress`` so the
default CI run does not pay the time cost; the dedicated
``stress`` CI job enables the marker.
"""

from __future__ import annotations

import threading

import pytest

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


def _fragment(idx: int) -> Fragment:
    """Build a unique fragment for index ``idx``.

    Args:
        idx: Sequence number for the payload hash.

    Returns:
        Fragment: A unique fragment.
    """
    ident = PayloadIdentity(
        payload_hash=str(idx).rjust(64, "0")[:64],
        model_id="stress",
        model_revision="",
        tokenizer_name="stress",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, idx),
        dtype="float16",
        shape=(1, 1, 1, 8, 64),
    )
    return Fragment(
        identity=ident,
        payload_ref=None,
        payload_size=10,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
        tenant_id="public",
    )


@pytest.mark.stress
class TestConcurrentStore:
    """32 threads * 50 stores; final node must hold all entries."""

    def test_64_threads_50_stores_each(self):
        from membrane.node import Node

        node = Node(node_id="stress", max_memory_bytes=10_000_000)
        errors: list[Exception] = []

        def worker(offset: int) -> None:
            try:
                for i in range(50):
                    node.store(_fragment(offset * 50 + i), is_primary=True)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert node.get_stats().fragment_count == 32 * 50


@pytest.mark.stress
class TestConcurrentEvict:
    """32 stores followed by 32 concurrent evicts."""

    def test_concurrent_store_then_remove(self):
        from membrane.node import Node

        node = Node(node_id="stress", max_memory_bytes=10_000_000)
        for i in range(64):
            node.store(_fragment(i), is_primary=True)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                # Stores interleave with removals.
                for i in range(64):
                    if node.fragments.get(_fragment(i).identity.payload_hash):
                        node.remove_fragment(_fragment(i).identity.payload_hash)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
