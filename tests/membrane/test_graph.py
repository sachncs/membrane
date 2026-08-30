from membrane.graph import Graph
from tests.conftest import make_fragment


def test_register_and_prefetch():
    g = Graph()
    g.add_node(make_fragment("h1"))
    g.add_edge("h1", "h2", "co_access")
    prefetch = g.prefetch_suggest("h1", edge_type="co_access")
    assert "h2" in prefetch


def test_suggest_prefetch_limit():
    g = Graph()
    for i in range(20):
        g.add_node(make_fragment(f"h{i}"))
        g.add_edge("root", f"h{i}", "co_access")
    prefetch = g.prefetch_suggest("root", limit=5)
    assert len(prefetch) == 5


def test_eviction_neighbors():
    g = Graph()
    g.add_node(make_fragment("h1"))
    g.add_edge("h1", "h2", "co_access")
    assert g.eviction_neighbors("h1") == {"h2"}
