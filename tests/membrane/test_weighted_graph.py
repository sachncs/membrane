"""Tests for weighted_graph module."""

import pytest

from membrane.weighted_graph import WeightedGraph


class TestWeightedGraph:
    """Test suite for WeightedGraph."""

    def test_add_weighted_edge_and_retrieve(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.9)
        assert g.get_edge_weight("a", "b", "next") == 0.9

    def test_get_edge_weight_missing(self):
        g = WeightedGraph()
        assert g.get_edge_weight("a", "b", "next") == 0.0

    def test_get_strong_neighbors(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.9)
        g.add_weighted_edge("a", "c", "next", 0.4)
        strong = g.get_strong_neighbors("a", edge_type="next", min_weight=0.5)
        assert strong == {"b"}

    def test_get_strong_neighbors_all_types(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.9)
        g.add_weighted_edge("a", "c", "prev", 0.8)
        strong = g.get_strong_neighbors("a", min_weight=0.5)
        assert strong == {"b", "c"}

    def test_has_node(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.5)
        assert g.has_node("a")
        assert g.has_node("b")
        assert not g.has_node("c")

    def test_multiple_edges_same_type(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.5)
        g.add_weighted_edge("a", "c", "next", 0.6)
        assert g.get_edge_weight("a", "b", "next") == 0.5
        assert g.get_edge_weight("a", "c", "next") == 0.6

    def test_update_weight_overwrites(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.5)
        g.add_weighted_edge("a", "b", "next", 0.9)
        assert g.get_edge_weight("a", "b", "next") == 0.9
