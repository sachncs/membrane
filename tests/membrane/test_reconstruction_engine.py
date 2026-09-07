from tests.conftest import make_fragment

"""Tests for Reconstructor."""

from membrane.fragment import Fragment
from membrane.fragmenter import compute_content_hash
from membrane.identity import PayloadIdentity
from membrane.index import Index
from membrane.prefilling import Adapter
from membrane.reconstructor import Reconstructor, ReconstructorConfig


def _fragment_for_span(tokens: list[int], start: int, end: int, model_id: str = "m") -> Fragment:
    """Build a fragment whose ``payload_hash`` matches the token slice.

    The reconstructor's exact-index lookup keys fragments by
    ``compute_content_hash(tokens)``; arbitrary placeholder hashes
    (``"a"``, ``"match"``, etc.) never resolve, so test fragments
    must derive their hash from the actual token sequence they
    cover.
    """
    window = tuple(tokens[start : end + 1])
    return make_fragment(
        compute_content_hash(window),
        (start, end),
        model_id=model_id,
    )


def test_full_exact_match_no_prefill():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter)
    tokens = list(range(100))
    frag = _fragment_for_span(tokens, 0, 99)
    index.insert(frag, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 1.0
    assert not result.prefill_invoked
    assert len(result.fragments) == 1


def test_partial_match_with_positional_extension():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter, config=ReconstructorConfig(max_gap_tokens=10))
    tokens = list(range(100))
    a = _fragment_for_span(tokens, 0, 39)
    b = _fragment_for_span(tokens, 40, 99)
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 1.0
    assert not result.prefill_invoked


def test_gap_filled_by_semantic_similarity():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter)
    tokens = list(range(100))

    gap_tokens = tuple(tokens[40:60])
    identity = PayloadIdentity(
        payload_hash=compute_content_hash(gap_tokens),
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(40, 59),
        dtype="float16",
        shape=(1, 1, 1, 20, 64),
    )
    gap_frag = Fragment(
        identity=identity,
        payload_ref=identity.payload_hash,
        payload_size=100,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )

    a = _fragment_for_span(tokens, 0, 39)
    b = _fragment_for_span(tokens, 60, 99)
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})
    index.insert(gap_frag, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 1.0


def test_large_gap_triggers_prefill():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter, config=ReconstructorConfig(max_gap_tokens=10))
    tokens = list(range(100))
    a = _fragment_for_span(tokens, 0, 19)
    b = _fragment_for_span(tokens, 80, 99)
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.prefill_invoked
    assert len(result.missing_segments) > 0


def test_empty_prompt():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter)
    result = engine.rebuild_context([], "m")
    assert result.coverage_ratio == 1.0
    assert result.fragments == []


def test_missing_index_triggers_prefill():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter, config=ReconstructorConfig(max_gap_tokens=5))
    tokens = list(range(50))
    result = engine.rebuild_context(tokens, "m")
    assert result.prefill_invoked


def test_coverage_ratio_accuracy():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter, config=ReconstructorConfig(max_gap_tokens=100))
    tokens = list(range(100))
    a = _fragment_for_span(tokens, 0, 49)
    index.insert(a, {"n1"})

    result = engine.rebuild_context(tokens, "m")
    assert result.coverage_ratio == 0.5


def test_graph_links_recorded():
    index = Index()
    adapter = Adapter()
    engine = Reconstructor(index, adapter, config=ReconstructorConfig(max_gap_tokens=10))
    tokens = list(range(50))
    a = _fragment_for_span(tokens, 0, 24)
    b = _fragment_for_span(tokens, 25, 49)
    index.insert(a, {"n1"})
    index.insert(b, {"n1"})

    engine.rebuild_context(tokens, "m")
    neighbors = index.co_access_neighbors(a.identity.payload_hash)
    assert b.identity.payload_hash in neighbors
