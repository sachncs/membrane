from tests.conftest import make_fragment

"""Tests for cluster_replicator module."""

import pytest

from membrane.fragment import Fragment
from membrane.node import Node
from membrane.replicator import Replicator
from membrane.signature import Signature


class TestClusterReplicator:
    """Test suite for Replicator."""

    def test_replicate_cluster_to_targets(self):
        cr = Replicator()
        source = Node("source")
        t1 = Node("t1")
        t2 = Node("t2")
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
        cr = Replicator()
        source = Node("source")
        t1 = Node("t1", max_memory_bytes=15)
        f1 = make_fragment("a", size=10)
        f2 = make_fragment("b", size=10)
        source.store(f1, is_primary=True)
        source.store(f2, is_primary=True)
        results = cr.replicate_cluster({"a", "b"}, source, [t1])
        assert len(results["t1"]) >= 1

    def test_replicate_cluster_empty_component(self):
        cr = Replicator()
        source = Node("source")
        t1 = Node("t1")
        results = cr.replicate_cluster(set(), source, [t1])
        assert results["t1"] == []
