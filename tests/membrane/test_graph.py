from membrane.fragment import Fragment
from membrane.graph import Graph
from membrane.signature import Signature


def test_register_and_prefetch():
    g = Graph()
    sig = Signature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    g.add_node(frag)
    g.add_edge("h1", "h2", "co_access")
    prefetch = g.prefetch_suggest("h1", edge_type="co_access")
    assert "h2" in prefetch


def test_suggest_prefetch_limit():
    g = Graph()
    sig = Signature("m", (0, 1), (0, 10))
    for i in range(20):
        g.add_node(Fragment(f"h{i}", (0.1,), sig, 10, 60.0, 0.5, 1))
        g.add_edge("root", f"h{i}", "co_access")
    prefetch = g.prefetch_suggest("root", limit=5)
    assert len(prefetch) == 5


def test_eviction_neighbors():
    g = Graph()
    sig = Signature("m", (0, 1), (0, 10))
    g.add_node(Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1))
    g.add_edge("h1", "h2", "co_access")
    assert g.eviction_neighbors("h1") == {"h2"}
