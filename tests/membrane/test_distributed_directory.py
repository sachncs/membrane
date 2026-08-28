"""Tests for distributed_directory module."""

import pytest

from membrane.directory import Directory
from membrane.ring import Ring
from membrane.supernode import Supernode


class TestDistributedDirectory:
    """Test suite for Directory."""

    def test_locate_across_supernodes(self):
        dd = Directory()
        sn1 = Supernode("sn1")
        sn1.register_fragment("h", "n1")
        sn2 = Supernode("sn2")
        sn2.register_fragment("h", "n2")
        dd.register_supernode(sn1)
        dd.register_supernode(sn2)
        assert dd.locate("h") == {"n1", "n2"}

    def test_locate_empty_returns_empty(self):
        dd = Directory()
        assert dd.locate("h") == set()

    def test_locate_nearest_from_holders(self):
        dd = Directory()
        sn = Supernode("sn1")
        sn.register_fragment("h", "n-b")
        sn.register_fragment("h", "n-a")
        dd.register_supernode(sn)
        nearest = dd.locate_nearest("h", "from-node")
        assert nearest == "n-a"

    def test_locate_nearest_falls_back_to_ring(self):
        ring = Ring()
        ring.add_node("ring-node")
        dd = Directory(hash_ring=ring)
        assert dd.locate_nearest("h", "from") == "ring-node"

    def test_add_node_propagates_to_supernodes(self):
        dd = Directory()
        sn = Supernode("sn1")
        dd.register_supernode(sn)
        dd.add_node("n1")
        assert sn.hash_ring.get_node("x") == "n1"

    def test_multiple_supernodes_aggregate(self):
        dd = Directory()
        sn1 = Supernode("sn1")
        sn1.register_fragment("h", "n1")
        dd.register_supernode(sn1)
        sn2 = Supernode("sn2")
        sn2.register_fragment("h", "n2")
        dd.register_supernode(sn2)
        assert dd.locate("h") == {"n1", "n2"}
