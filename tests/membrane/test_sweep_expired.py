"""The periodic sweeper must only remove expired fragments.

It used to call ``Node.evict``, whose LRU / graph phases removed live
fragments every sweep interval.
"""

from __future__ import annotations

from membrane.fragment import Fragment
from membrane.node import Node
from tests.conftest import make_fragment


def _frag(content_hash: str, ttl: float) -> Fragment:
    base = make_fragment(content_hash)
    return Fragment(
        identity=base.identity,
        payload_ref=None,
        payload_size=10,
        ttl=ttl,
        reuse_score=0.5,
        version_id=1,
    )


def test_sweep_expired_keeps_live_fragments() -> None:
    node = Node("n1", max_memory_bytes=10_000)
    assert node.store(_frag("live", ttl=3600.0))
    assert node.store(_frag("stale", ttl=1.0))
    evicted_seen: list[object] = []
    node.add_eviction_callback(evicted_seen.append)

    now = node.insertion_times["stale"] + 5.0
    assert node.sweep_expired(current_time=now) == ["stale"]
    assert set(node.fragments) == {"live"}
    assert len(evicted_seen) == 1
    # A second sweep with nothing expired is a no-op.
    assert node.sweep_expired(current_time=now) == []
    assert set(node.fragments) == {"live"}
