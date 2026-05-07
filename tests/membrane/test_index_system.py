from membrane.fragment import Fragment
from membrane.index_system import IndexSystem
from membrane.structural_signature import StructuralSignature


def test_cross_index_query():
    sys = IndexSystem()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1, 0.0, 0.0), sig, 10, 60.0, 0.5, 1)
    sys.insert(frag, {"node-a"})

    exact = sys.exact_lookup("h1")
    assert exact is not None
    assert exact.fragment == frag
    assert "node-a" in exact.locations

    semantic = sys.semantic_lookup((0.1, 0.0, 0.0), k=1)
    assert semantic[0].content_hash == "h1"


def test_batch_insert_and_query():
    sys = IndexSystem()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    for i in range(5):
        frag = Fragment(f"h{i}", (float(i), 0.0, 0.0), sig, 10, 60.0, 0.5, 1)
        sys.insert(frag, {"node-a"})

    assert len(sys.semantic_lookup((0.0, 0.0, 0.0), k=3)) == 3
    assert sys.exact_lookup("h2") is not None
    assert len(sys.positional_lookup(0, 15)) == 5


def test_co_access_through_facade():
    sys = IndexSystem()
    sys.record_co_access("a", "b")
    assert sys.co_access_neighbors("a") == {"b"}
