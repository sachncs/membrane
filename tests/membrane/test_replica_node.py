"""Tests for replica_node module."""

import pytest

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.origin_node import OriginNode
from membrane.replica_node import ReplicaNode
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash, size=100):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0, 0.0),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 1)
        ),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestReplicaNode:
    """Test suite for ReplicaNode."""

    def test_warm_from_origin(self):
        origin = OriginNode("origin-1")
        replica = ReplicaNode("replica-1")
        f1 = make_fragment("a", size=40)
        f2 = make_fragment("b", size=40)
        origin.store(f1, is_primary=True)
        origin.store(f2, is_primary=True)
        warmed = replica.warm_from_origin(origin, ["a", "b"])
        assert "a" in warmed
        assert "b" in warmed
        assert replica.retrieve("a") is not None

    def test_warm_missing_hash_returns_empty(self):
        origin = OriginNode("origin-1")
        replica = ReplicaNode("replica-1")
        warmed = replica.warm_from_origin(origin, ["missing"])
        assert warmed == []

    def test_store_always_non_primary(self):
        replica = ReplicaNode("r")
        frag = make_fragment("x", size=10)
        replica.store(frag, is_primary=True)
        assert "x" not in replica.primary_hashes

    def test_replica_is_membrane_node_subclass(self):
        replica = ReplicaNode("r")
        assert isinstance(replica, MembraneNode)

    def test_warm_partial_success(self):
        origin = OriginNode("origin-1")
        replica = ReplicaNode("replica-1", max_memory_bytes=50)
        f1 = make_fragment("a", size=30)
        f2 = make_fragment("b", size=30)
        origin.store(f1, is_primary=True)
        origin.store(f2, is_primary=True)
        warmed = replica.warm_from_origin(origin, ["a", "b"])
        assert len(warmed) >= 1
