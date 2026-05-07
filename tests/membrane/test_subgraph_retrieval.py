"""Tests for subgraph_retrieval module."""

import pytest

from membrane.subgraph_retrieval import SubgraphRetrieval
from membrane.weighted_graph import WeightedGraph


def build_graph():
    g = WeightedGraph()
    g.add_weighted_edge("a", "b", "next", 0.9)
    g.add_weighted_edge("b", "c", "next", 0.9)
    g.add_weighted_edge("a", "d", "next", 0.3)
    return g


class TestSubgraphRetrieval:
    """Test suite for SubgraphRetrieval."""

    def test_retrieve_component_depth_1(self):
        g = build_graph()
        sr = SubgraphRetrieval(g)
        comp = sr.retrieve_component("a", min_weight=0.5, max_depth=1)
        assert comp == {"a", "b"}

    def test_retrieve_component_depth_2(self):
        g = build_graph()
        sr = SubgraphRetrieval(g)
        comp = sr.retrieve_component("a", min_weight=0.5, max_depth=2)
        assert comp == {"a", "b", "c"}

    def test_retrieve_component_weak_edges_ignored(self):
        g = build_graph()
        sr = SubgraphRetrieval(g)
        comp = sr.retrieve_component("a", min_weight=0.5, max_depth=3)
        assert "d" not in comp

    def test_retrieve_component_missing_node(self):
        g = build_graph()
        sr = SubgraphRetrieval(g)
        comp = sr.retrieve_component("z", min_weight=0.5, max_depth=3)
        assert comp == set()

    def test_retrieve_clusters(self):
        g = build_graph()
        sr = SubgraphRetrieval(g)
        clusters = sr.retrieve_clusters(["a", "c"], min_weight=0.5, max_depth=2)
        assert len(clusters) == 1
        assert clusters[0] == {"a", "b", "c"}

    def test_retrieve_clusters_disjoint(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.9)
        g.add_weighted_edge("x", "y", "next", 0.9)
        sr = SubgraphRetrieval(g)
        clusters = sr.retrieve_clusters(["a", "x"], min_weight=0.5, max_depth=1)
        assert len(clusters) == 2
        assert {"a", "b"} in clusters
        assert {"x", "y"} in clusters
