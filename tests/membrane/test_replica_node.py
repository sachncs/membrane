"""Tests for replica_node module."""

import pytest

from membrane.fragment import Fragment
from membrane.node import Node
from membrane.origin import Origin
from membrane.replica import Replica
from membrane.signature import Signature

class TestReplicaNode:
    """Test suite for Replica."""

    def test_warm_from_origin(self):
        origin = Origin("origin-1")
        replica = Replica("replica-1")
        f1 = make_fragment("a", size=40)
        f2 = make_fragment("b", size=40)
        origin.store(f1, is_primary=True)
        origin.store(f2, is_primary=True)
        warmed = replica.warm_from_origin(origin, ["a", "b"])
        assert "a" in warmed
        assert "b" in warmed
        assert replica.retrieve("a") is not None

    def test_warm_missing_hash_returns_empty(self):
        origin = Origin("origin-1")
        replica = Replica("replica-1")
        warmed = replica.warm_from_origin(origin, ["missing"])
        assert warmed == []

    def test_store_always_non_primary(self):
        replica = Replica("r")
        frag = make_fragment("x", size=10)
        replica.store(frag, is_primary=True)
        assert "x" not in replica.primary_hashes

    def test_replica_is_membrane_node_subclass(self):
        replica = Replica("r")
        assert isinstance(replica, Node)

    def test_warm_partial_success(self):
        origin = Origin("origin-1")
        replica = Replica("replica-1", max_memory_bytes=50)
        f1 = make_fragment("a", size=30)
        f2 = make_fragment("b", size=30)
        origin.store(f1, is_primary=True)
        origin.store(f2, is_primary=True)
        warmed = replica.warm_from_origin(origin, ["a", "b"])
        assert len(warmed) >= 1
