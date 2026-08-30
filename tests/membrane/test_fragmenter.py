"""Tests for Fragmenter."""

from membrane.fragment import Fragment
from membrane.fragmenter import (
    Fragmenter,
    FragmenterConfig,
    compute_content_hash,
    generate_embedding,
)
from membrane.identity import PayloadIdentity


def test_create_windows_produces_correct_spans():
    engine = Fragmenter(FragmenterConfig(window_size=4))
    tokens = list(range(10))
    frags = engine.create_windows(tokens, model_id="m")
    assert len(frags) == 3
    assert frags[0].identity.token_span == (0, 3)
    assert frags[1].identity.token_span == (4, 7)
    assert frags[2].identity.token_span == (8, 9)


def test_create_windows_empty_prompt():
    engine = Fragmenter()
    assert engine.create_windows([], model_id="m") == []


def test_split_preserves_coverage():
    engine = Fragmenter(FragmenterConfig(window_size=8))
    tokens = list(range(8))
    frags = engine.create_windows(tokens, model_id="m")
    parent = frags[0]
    children = engine.split(parent, [3])
    assert len(children) == 2
    assert children[0].identity.token_span == (0, 3)
    assert children[1].identity.token_span == (4, 7)


def test_split_generates_new_hashes():
    engine = Fragmenter(FragmenterConfig(window_size=4))
    tokens = list(range(4))
    frags = engine.create_windows(tokens, model_id="m")
    parent = frags[0]
    children = engine.split(parent, [1, 2])
    hashes = {c.identity.payload_hash for c in children}
    assert len(hashes) == 3
    assert parent.identity.payload_hash not in hashes


def test_merge_combines_adjacent():
    engine = Fragmenter(FragmenterConfig(window_size=4))
    tokens = list(range(8))
    frags = engine.create_windows(tokens, model_id="m")
    merged = engine.merge(frags)
    assert merged is not None
    assert merged.identity.token_span == (0, 7)


def test_merge_rejects_non_adjacent():
    engine = Fragmenter(FragmenterConfig(window_size=4))
    a = engine.create_windows(list(range(4)), model_id="m")[0]
    b = engine.create_windows(list(range(10, 14)), model_id="m")[0]
    assert engine.merge([a, b]) is None


def test_merge_rejects_different_model():
    engine = Fragmenter(FragmenterConfig(window_size=4))
    a = engine.create_windows(list(range(4)), model_id="m1")[0]
    b = engine.create_windows(list(range(4, 8)), model_id="m2")[0]
    assert engine.merge([a, b]) is None


def test_merge_rejects_high_reuse():
    engine = Fragmenter(FragmenterConfig(window_size=4))
    a = engine.create_windows(list(range(4)), model_id="m")[0]
    b = engine.create_windows(list(range(4, 8)), model_id="m")[0]
    # Manually create high-reuse copies using the new schema.
    high_a = Fragment(
        identity=a.identity,
        payload_ref=a.payload_ref,
        payload_size=a.payload_size,
        ttl=a.ttl,
        reuse_score=0.9,
        version_id=a.version_id,
    )
    high_b = Fragment(
        identity=b.identity,
        payload_ref=b.payload_ref,
        payload_size=b.payload_size,
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


def test_payload_identity_round_trip():
    identity = PayloadIdentity(
        payload_hash="abc",
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 3),
        dtype="float16",
        shape=(1, 1, 1, 4, 64),
    )
    assert identity.fingerprint() == identity.fingerprint()
