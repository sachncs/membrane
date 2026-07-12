"""Tests for cluster_replicator module."""

import pytest

from membrane.cluster_replicator import ClusterReplicator
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash, size=10):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestClusterReplicator:
    """Test suite for ClusterReplicator."""

    def test_replicate_cluster_to_targets(self):
        cr = ClusterReplicator()
        source = MembraneNode("source")
        t1 = MembraneNode("t1")
        t2 = MembraneNode("t2")
        f1 = make_fragment("a", size=10)
        f2 = make_fragment("b", size=10)
        source.store(f1, is_primary=True)
        source.store(f2, is_primary=True)
        results = cr.replicate_cluster({"a", "b"}, source, [t1, t2])
        assert set(results["t1"]) == {"a", "b"}
        assert set(results["t2"]) == {"a", "b"}
        assert t1.retrieve("a") is not None
        assert t2.retrieve("b") is not None

    def test_replicate_cluster_partial(self):
        cr = ClusterReplicator()
        source = MembraneNode("source")
        t1 = MembraneNode("t1", max_memory_bytes=15)
        f1 = make_fragment("a", size=10)
        f2 = make_fragment("b", size=10)
        source.store(f1, is_primary=True)
        source.store(f2, is_primary=True)
        results = cr.replicate_cluster({"a", "b"}, source, [t1])
        assert len(results["t1"]) >= 1

    def test_replicate_cluster_empty_component(self):
        cr = ClusterReplicator()
        source = MembraneNode("source")
        t1 = MembraneNode("t1")
        results = cr.replicate_cluster(set(), source, [t1])
        assert results["t1"] == []
