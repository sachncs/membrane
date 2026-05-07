"""Tests for ReconstructionEngine."""

from membrane.fragment import Fragment
from membrane.fragmentation_engine import FragmentationEngine
from membrane.index_system import IndexSystem
from membrane.prefill_adapter import PrefillAdapter
from membrane.reconstruction_engine import ReconstructionConfig, ReconstructionEngine
from membrane.structural_signature import StructuralSignature


def make_fragment(
    content_hash: str, token_span: tuple[int, int], model_id: str = "m"
) -> Fragment:
    sig = StructuralSignature(model_id, (0, 1), token_span)
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2, 0.3),
        structural_signature=sig,
        size=100,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


def test_full_exact_match_no_prefill():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter)
    tokens = list(range(100))
    frag = make_fragment("match", (0, 99))
    index.insert(frag, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 1.0
    assert not result.prefill_invoked
    assert len(result.fragments) == 1


def test_partial_match_with_positional_extension():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter, config=ReconstructionConfig(max_gap_tokens=10))
    tokens = list(range(100))
    a = make_fragment("a", (0, 39))
    b = make_fragment("b", (40, 99))
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 1.0
    assert not result.prefill_invoked


def test_gap_filled_by_semantic_similarity():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter)
    tokens = list(range(100))

    from membrane.fragmentation_engine import generate_embedding

    gap_tokens = tuple(tokens[40:60])
    gap_embedding = generate_embedding(gap_tokens, 128)
    gap_frag = Fragment(
        content_hash="gap",
        embedding=gap_embedding,
        structural_signature=StructuralSignature("m", (0, 1), (40, 59)),
        size=100,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )

    a = make_fragment("a", (0, 39))
    b = make_fragment("b", (60, 99))
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})
    index.insert(gap_frag, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 1.0


def test_large_gap_triggers_prefill():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter, config=ReconstructionConfig(max_gap_tokens=10))
    tokens = list(range(100))
    a = make_fragment("a", (0, 19))
    b = make_fragment("b", (80, 99))
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.prefill_invoked
    assert len(result.missing_segments) > 0


def test_empty_prompt():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter)
    result = engine.rebuild_context([], "m")
    assert result.coverage_ratio == 1.0
    assert result.fragments == []


def test_missing_index_triggers_prefill():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter, config=ReconstructionConfig(max_gap_tokens=5))
    tokens = list(range(50))
    result = engine.rebuild_context(tokens, "m")
    assert result.prefill_invoked


def test_coverage_ratio_accuracy():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter, config=ReconstructionConfig(max_gap_tokens=100))
    tokens = list(range(100))
    a = make_fragment("a", (0, 49))
    index.insert(a, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 0.5


def test_graph_links_recorded():
    index = IndexSystem()
    adapter = PrefillAdapter()
    engine = ReconstructionEngine(index, adapter, config=ReconstructionConfig(max_gap_tokens=10))
    tokens = list(range(50))
    a = make_fragment("a", (0, 24))
    b = make_fragment("b", (25, 49))
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})

    engine.rebuild_context(tokens, "m")
    neighbors = index.co_access_neighbors("a")
    assert "b" in neighbors
