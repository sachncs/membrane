"""Tests for semantic_cluster module."""

import pytest

from membrane.fragment import Fragment
from membrane.semantic_cluster import SemanticCluster
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash, embedding):
    return Fragment(
        content_hash=content_hash,
        embedding=embedding,
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=10,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestSemanticCluster:
    """Test suite for SemanticCluster."""

    def test_emptyfragments(self):
        sc = SemanticCluster()
        assert sc.cluster([]) == []

    def test_singlefragment_one_cluster(self):
        sc = SemanticCluster()
        frag = make_fragment("a", (1.0, 0.0))
        clusters = sc.cluster([frag], similarity_threshold=0.9)
        assert len(clusters) == 1
        assert clusters[0] == [frag]

    def test_identical_embeddings_clustered(self):
        sc = SemanticCluster()
        f1 = make_fragment("a", (1.0, 0.0))
        f2 = make_fragment("b", (1.0, 0.0))
        clusters = sc.cluster([f1, f2], similarity_threshold=0.99)
        assert len(clusters) == 1
        assert set(clusters[0]) == {f1, f2}

    def test_different_embeddings_separate_clusters(self):
        sc = SemanticCluster()
        f1 = make_fragment("a", (1.0, 0.0))
        f2 = make_fragment("b", (0.0, 1.0))
        clusters = sc.cluster([f1, f2], similarity_threshold=0.9)
        assert len(clusters) == 2

    def test_threshold_controls_clustering(self):
        sc = SemanticCluster()
        f1 = make_fragment("a", (1.0, 0.0))
        f2 = make_fragment("b", (0.8, 0.2))
        clusters_loose = sc.cluster([f1, f2], similarity_threshold=0.8)
        clusters_tight = sc.cluster([f1, f2], similarity_threshold=0.99)
        assert len(clusters_loose) == 1
        assert len(clusters_tight) == 2

    def test_cosine_similarity_orthogonal(self):
        assert SemanticCluster.cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0

    def test_cosine_similarity_identical(self):
        assert SemanticCluster.cosine_similarity((1.0, 2.0), (1.0, 2.0)) == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector(self):
        assert SemanticCluster.cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0
