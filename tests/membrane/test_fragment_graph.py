from membrane.fragment import Fragment
from membrane.fragment_graph import FragmentGraph
from membrane.structural_signature import StructuralSignature


def test_add_node_and_edge():
    g = FragmentGraph()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    g.add_node(frag)
    g.add_edge("h1", "h2", edge_type="co_access")
    assert g.has_node("h1")
    assert g.has_edge("h1", "h2", "co_access")


def test_neighbors_by_type():
    g = FragmentGraph()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    g.add_node(Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1))
    g.add_edge("h1", "h2", "co_access")
    g.add_edge("h1", "h3", "semantic")
    assert g.neighbors("h1", "co_access") == {"h2"}
    assert g.neighbors("h1", "semantic") == {"h3"}
    assert g.neighbors("h1") == {"h2", "h3"}


def test_get_fragment():
    g = FragmentGraph()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    g.add_node(frag)
    assert g.get_fragment("h1") == frag
    assert g.get_fragment("missing") is None
