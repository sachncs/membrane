"""Tests for semantic_hash module."""

from membrane.semantic_hash import compute_semantic_hash, semantic_distance


def test_compute_semantic_hash_is_deterministic():
    emb = (0.5, -0.5, 0.0)
    a = compute_semantic_hash(emb, precision=8)
    b = compute_semantic_hash(emb, precision=8)
    assert a == b


def test_nearby_embeddings_share_hash():
    emb_a = (0.501, -0.499, 0.001)
    emb_b = (0.500, -0.500, 0.000)
    hash_a = compute_semantic_hash(emb_a, precision=8)
    hash_b = compute_semantic_hash(emb_b, precision=8)
    assert hash_a == hash_b


def test_distant_embeddings_differ():
    emb_a = (1.0, 1.0, 1.0)
    emb_b = (-1.0, -1.0, -1.0)
    hash_a = compute_semantic_hash(emb_a, precision=4)
    hash_b = compute_semantic_hash(emb_b, precision=4)
    assert hash_a != hash_b


def test_semantic_distance_identical():
    h = compute_semantic_hash((0.1, 0.2), precision=8)
    assert semantic_distance(h, h) == 0.0


def test_semantic_distance_different():
    h_a = compute_semantic_hash((1.0, 0.0), precision=8)
    h_b = compute_semantic_hash((-1.0, 0.0), precision=8)
    d = semantic_distance(h_a, h_b)
    assert 0.0 < d <= 1.0


def test_empty_embedding():
    assert compute_semantic_hash((), precision=8) == "0"
