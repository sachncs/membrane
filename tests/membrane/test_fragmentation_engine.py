"""Tests for FragmentationEngine."""

from membrane.fragmentation_engine import (
    FragmentationConfig,
    FragmentationEngine,
    compute_content_hash,
    generate_embedding,
)
from membrane.structural_signature import StructuralSignature


def test_create_windows_produces_correct_spans():
    engine = FragmentationEngine(FragmentationConfig(window_size=4))
    tokens = list(range(10))
    frags = engine.create_windows(tokens, model_id="m")
    assert len(frags) == 3
    assert frags[0].structural_signature.token_span == (0, 3)
    assert frags[1].structural_signature.token_span == (4, 7)
    assert frags[2].structural_signature.token_span == (8, 9)


def test_create_windows_empty_prompt():
    engine = FragmentationEngine()
    assert engine.create_windows([], model_id="m") == []


def test_split_preserves_coverage():
    engine = FragmentationEngine(FragmentationConfig(window_size=8))
    tokens = list(range(8))
    frags = engine.create_windows(tokens, model_id="m")
    parent = frags[0]
    children = engine.split(parent, [3])
    assert len(children) == 2
    assert children[0].structural_signature.token_span == (0, 3)
    assert children[1].structural_signature.token_span == (4, 7)


def test_split_generates_new_hashes():
    engine = FragmentationEngine(FragmentationConfig(window_size=4))
    tokens = list(range(4))
    frags = engine.create_windows(tokens, model_id="m")
    parent = frags[0]
    children = engine.split(parent, [1, 2])
    hashes = {c.content_hash for c in children}
    assert len(hashes) == 3
    assert parent.content_hash not in hashes


def test_merge_combines_adjacent():
    engine = FragmentationEngine(FragmentationConfig(window_size=4))
    tokens = list(range(8))
    frags = engine.create_windows(tokens, model_id="m")
    merged = engine.merge(frags)
    assert merged is not None
    assert merged.structural_signature.token_span == (0, 7)


def test_merge_rejects_non_adjacent():
    engine = FragmentationEngine(FragmentationConfig(window_size=4))
    a = engine.create_windows(list(range(4)), model_id="m")[0]
    b = engine.create_windows(list(range(10, 14)), model_id="m")[0]
    assert engine.merge([a, b]) is None


def test_merge_rejects_different_model():
    engine = FragmentationEngine(FragmentationConfig(window_size=4))
    a = engine.create_windows(list(range(4)), model_id="m1")[0]
    b = engine.create_windows(list(range(4, 8)), model_id="m2")[0]
    assert engine.merge([a, b]) is None


def test_merge_rejects_high_reuse():
    engine = FragmentationEngine(FragmentationConfig(window_size=4))
    a = engine.create_windows(list(range(4)), model_id="m")[0]
    b = engine.create_windows(list(range(4, 8)), model_id="m")[0]
    # Manually create high-reuse copies
    high_a = a.__class__(
        content_hash=a.content_hash,
        embedding=a.embedding,
        structural_signature=a.structural_signature,
        size=a.size,
        ttl=a.ttl,
        reuse_score=0.9,
        version_id=a.version_id,
    )
    high_b = b.__class__(
        content_hash=b.content_hash,
        embedding=b.embedding,
        structural_signature=b.structural_signature,
        size=b.size,
        ttl=b.ttl,
        reuse_score=0.9,
        version_id=b.version_id,
    )
    assert engine.merge([high_a, high_b]) is None


def test_content_hash_is_deterministic():
    a = compute_content_hash((1, 2, 3))
    b = compute_content_hash((1, 2, 3))
    assert a == b
    c = compute_content_hash((1, 2, 4))
    assert a != c


def test_embedding_is_normalized():
    emb = generate_embedding((1, 2, 3), dim=64)
    norm = sum(v * v for v in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-6
