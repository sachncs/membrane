"""Tests for supernode module."""

import pytest

from membrane.hash_ring import HashRing
from membrane.supernode import Supernode


class TestSupernode:
    """Test suite for Supernode."""

    def test_register_and_resolve(self):
        sn = Supernode("sn-1")
        sn.register_fragment("hash-a", "node-1")
        sn.register_fragment("hash-a", "node-2")
        assert sn.resolve("hash-a") == {"node-1", "node-2"}

    def test_resolve_unknown_returns_empty(self):
        sn = Supernode("sn-1")
        assert sn.resolve("missing") == set()

    def test_unregister_fragment(self):
        sn = Supernode("sn-1")
        sn.register_fragment("h", "n1")
        sn.unregister_fragment("h", "n1")
        assert sn.resolve("h") == set()
        assert "h" not in sn.fragment_locations

    def test_resolve_via_ring(self):
        ring = HashRing()
        ring.add_node("node-a")
        sn = Supernode("sn-1", hash_ring=ring)
        assert sn.resolve_via_ring("abc") == "node-a"

    def test_add_node_to_ring(self):
        sn = Supernode("sn-1")
        sn.add_node("node-a")
        assert sn.hash_ring.get_node("x") == "node-a"

    def test_remove_node_from_ring(self):
        from membrane.hash_ring import EmptyRingError

        sn = Supernode("sn-1")
        sn.add_node("node-a")
        sn.remove_node("node-a")
        with pytest.raises(EmptyRingError):
            sn.hash_ring.get_node("x")
