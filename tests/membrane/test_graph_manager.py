from membrane.fragment import Fragment
from membrane.graph_manager import GraphManager
from membrane.structural_signature import StructuralSignature


def test_register_and_prefetch():
    gm = GraphManager()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    gm.register(frag)
    gm.link("h1", "h2", "co_access")
    prefetch = gm.suggest_prefetch("h1", edge_type="co_access")
    assert "h2" in prefetch


def test_suggest_prefetch_limit():
    gm = GraphManager()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    for i in range(20):
        gm.register(Fragment(f"h{i}", (0.1,), sig, 10, 60.0, 0.5, 1))
        gm.link("root", f"h{i}", "co_access")
    prefetch = gm.suggest_prefetch("root", limit=5)
    assert len(prefetch) == 5


def test_eviction_candidates():
    gm = GraphManager()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    gm.register(Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1))
    gm.link("h1", "h2", "co_access")
    assert gm.eviction_candidates("h1") == {"h2"}
