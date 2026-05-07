"""Tests for PrefillAdapter."""

from membrane.model.router import Router
from membrane.prefill_adapter import PrefillAdapter


def test_prefill_returns_reasonable_kv_size():
    adapter = PrefillAdapter()
    result = adapter.prefill(list(range(1024)), model_id="m")
    assert result.kv_size_mib > 0.0
    assert result.latency_seconds > 0.0


def test_prefill_uses_router():
    router = Router(threshold=512)
    adapter = PrefillAdapter(router=router)
    result = adapter.prefill(list(range(1024)), model_id="m")
    assert result.routing_decision is not None
    assert result.routing_decision.target in ("membrane", "pd-p")


def test_prefill_no_router_skips_decision():
    adapter = PrefillAdapter()
    result = adapter.prefill(list(range(1024)), model_id="m")
    assert result.routing_decision is None


def test_convert_kv_to_fragments_produces_fragments():
    adapter = PrefillAdapter()
    tokens = list(range(100))
    frags = adapter.convert_kv_to_fragments(tokens, "m", kv_size_mib=10.0)
    assert len(frags) > 0
    assert all(isinstance(f.content_hash, str) for f in frags)
    assert sum(f.size for f in frags) > 0


def test_empty_prompt_returns_empty():
    adapter = PrefillAdapter()
    result = adapter.prefill([], model_id="m")
    assert result.fragments == []
    # profiler clamps length 0 to the 1024 boundary
    assert result.kv_size_mib == 190.8


def test_very_long_prompt_clamps():
    adapter = PrefillAdapter()
    result = adapter.prefill(list(range(200_000)), model_id="m")
    # profiler clamps to 131072 boundary
    assert result.kv_size_mib > 0.0
    assert result.latency_seconds > 0.0


def test_fragments_cover_full_prompt():
    adapter = PrefillAdapter()
    tokens = list(range(500))
    frags = adapter.convert_kv_to_fragments(tokens, "m", kv_size_mib=5.0)
    spans = [f.structural_signature.token_span for f in frags]
    assert spans[0][0] == 0
    assert spans[-1][1] == 499
    # Adjacency: each span starts right after the previous ends
    for i in range(1, len(spans)):
        assert spans[i][0] == spans[i - 1][1] + 1
