"""Tests for SemanticCluster."""

import pytest

from membrane.semantics import SemanticCluster, cosine_similarity
from tests.conftest import make_fragment


class TestSemanticCluster:
    """Test suite for SemanticCluster."""

    def test_empty_fragments(self):
        sc = SemanticCluster()
        assert sc.cluster([]) == []

    def test_single_fragment_one_cluster(self):
        sc = SemanticCluster()
        frag = make_fragment("a", (1.0, 0.0))
        clusters = sc.cluster([frag], similarity_threshold=0.9)
        assert len(clusters) == 1
        assert clusters[0] == [frag]

    def test_identical_payload_hashes_clustered(self):
        """Under the new schema, clustering is by ``payload_hash``.

        Fragments that share a payload hash are byte-identical and
        collapse into a single cluster regardless of ``similarity_threshold``.
        """
        sc = SemanticCluster()
        f1 = make_fragment("a", (1.0, 0.0))
        f2 = make_fragment("a", (1.0, 0.0))
        clusters = sc.cluster([f1, f2], similarity_threshold=0.99)
        assert len(clusters) == 1
        assert set(clusters[0]) == {f1, f2}

    def test_different_payload_hashes_separate_clusters(self):
        sc = SemanticCluster()
        f1 = make_fragment("a", (1.0, 0.0))
        f2 = make_fragment("b", (0.0, 1.0))
        clusters = sc.cluster([f1, f2], similarity_threshold=0.9)
        assert len(clusters) == 2

    def test_threshold_does_not_affect_hash_clustering(self):
        sc = SemanticCluster()
        f1 = make_fragment("a", (1.0, 0.0))
        f2 = make_fragment("b", (0.8, 0.2))
        clusters_loose = sc.cluster([f1, f2], similarity_threshold=0.8)
        clusters_tight = sc.cluster([f1, f2], similarity_threshold=0.99)
        # Distinct hashes — never clustered together.
        assert len(clusters_loose) == 2
        assert len(clusters_tight) == 2

    def test_cosine_similarity_orthogonal(self):
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0

    def test_cosine_similarity_identical(self):
        assert cosine_similarity((1.0, 2.0), (1.0, 2.0)) == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector(self):
        assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0
