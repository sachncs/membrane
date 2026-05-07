"""Tests for hash_ring module."""

import pytest

from membrane.hash_ring import EmptyRingError, HashRing


class TestHashRing:
    """Test suite for HashRing."""

    def test_get_node_empty_ring_raises(self):
        """An empty ring is a configuration error; should raise."""
        ring = HashRing()
        with pytest.raises(EmptyRingError):
            ring.get_node("abc")

    def test_get_nodes_empty_ring_raises(self):
        ring = HashRing()
        with pytest.raises(EmptyRingError):
            ring.get_nodes("abc")

    def test_add_node_enables_lookup(self):
        ring = HashRing()
        ring.add_node("node-a")
        node = ring.get_node("abc")
        assert node == "node-a"

    def test_multiple_nodes_distribute(self):
        ring = HashRing()
        ring.add_node("node-a")
        ring.add_node("node-b")
        results = {ring.get_node(str(i)) for i in range(100)}
        assert len(results) == 2

    def test_remove_node(self):
        ring = HashRing()
        ring.add_node("node-a")
        ring.add_node("node-b")
        ring.remove_node("node-a")
        for i in range(100):
            assert ring.get_node(str(i)) == "node-b"

    def test_remove_unknown_node_is_noop(self):
        ring = HashRing()
        ring.remove_node("ghost")
        assert ring.sorted_keys == []

    def test_add_duplicate_is_noop(self):
        ring = HashRing()
        ring.add_node("node-a")
        ring.add_node("node-a")
        assert len(ring.node_ids) == 1
        assert len(ring.sorted_keys) == 150

    def test_get_nodes_returns_top_n(self):
        ring = HashRing()
        for n in ["a", "b", "c"]:
            ring.add_node(n)
        nodes = ring.get_nodes("key", n=2)
        assert len(nodes) == 2
        assert len(set(nodes)) == 2

    def test_get_nodes_more_than_available(self):
        ring = HashRing()
        ring.add_node("a")
        nodes = ring.get_nodes("key", n=5)
        assert nodes == ["a"]

    def test_consistent_hashing_stable(self):
        ring = HashRing()
        ring.add_node("node-a")
        ring.add_node("node-b")
        key = "stable-key"
        first = ring.get_node(key)
        for _ in range(10):
            assert ring.get_node(key) == first

    def test_virtual_nodes_create_many_points(self):
        ring = HashRing(virtual_nodes=10)
        ring.add_node("a")
        assert len(ring.sorted_keys) == 10

    def test_remove_all_nodes_leaves_empty_ring(self):
        ring = HashRing()
        ring.add_node("a")
        ring.remove_node("a")
        with pytest.raises(EmptyRingError):
            ring.get_node("x")

    def test_error_message_includes_hint(self):
        ring = HashRing()
        with pytest.raises(EmptyRingError, match="add at least one node"):
            ring.get_node("x")
