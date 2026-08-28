"""Tests for Adapter."""

from membrane.adapter import Adapter
from membrane.model.router import Router


def test_prefill_returns_reasonable_kv_size():
    adapter = Adapter()
    result = adapter.prefill(list(range(1024)), model_id="m")
    assert result.kv_size > 0.0
    assert result.latency_seconds > 0.0


def test_prefill_uses_router():
    router = Router(threshold=512)
    adapter = Adapter(router=router)
    result = adapter.prefill(list(range(1024)), model_id="m")
    assert result.routing_decision is not None
    assert result.routing_decision.target in ("membrane", "pd-p")


def test_prefill_no_router_skips_decision():
    adapter = Adapter()
    result = adapter.prefill(list(range(1024)), model_id="m")
    assert result.routing_decision is None


def test_convert_kv_to_fragments_produces_fragments():
    adapter = Adapter()
    tokens = list(range(100))
    frags = adapter.kv_fragments(tokens, "m", kv_size=10.0)
    assert len(frags) > 0
    assert all(isinstance(f.content_hash, str) for f in frags)
    assert sum(f.size for f in frags) > 0


def test_empty_prompt_returns_empty():
    adapter = Adapter()
    result = adapter.prefill([], model_id="m")
    assert result.fragments == []
    # profiler clamps length 0 to the 1024 boundary
    assert result.kv_size == 190.8


def test_very_long_prompt_clamps():
    adapter = Adapter()
    result = adapter.prefill(list(range(200_000)), model_id="m")
    # profiler clamps to 131072 boundary
    assert result.kv_size > 0.0
    assert result.latency_seconds > 0.0


def test_fragments_cover_full_prompt():
    adapter = Adapter()
    tokens = list(range(500))
    frags = adapter.kv_fragments(tokens, "m", kv_size=5.0)
    spans = [f.structural_signature.token_span for f in frags]
    assert spans[0][0] == 0
    assert spans[-1][1] == 499
    # Adjacency: each span starts right after the previous ends
    for i in range(1, len(spans)):
        assert spans[i][0] == spans[i - 1][1] + 1
