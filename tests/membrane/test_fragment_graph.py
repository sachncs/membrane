from membrane.graph import Graph
from tests.conftest import make_fragment


def test_add_node_and_edge():
    g = Graph()
    frag = make_fragment("h1")
    g.add_node(frag)
    g.add_edge("h1", "h2", edge_type="co_access")
    assert g.has_node("h1")
    assert g.has_edge("h1", "h2", "co_access")


def test_neighbors_by_type():
    g = Graph()
    g.add_node(make_fragment("h1"))
    g.add_edge("h1", "h2", "co_access")
    g.add_edge("h1", "h3", "semantic")
    assert g.neighbors("h1", "co_access") == {"h2"}
    assert g.neighbors("h1", "semantic") == {"h3"}
    assert g.neighbors("h1") == {"h2", "h3"}


def test_get_fragment():
    g = Graph()
    frag = make_fragment("h1")
    g.add_node(frag)
    assert g.get_fragment("h1") == frag
    assert g.get_fragment("missing") is None
