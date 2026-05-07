from membrane.exact_index import ExactIndex
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


def test_index_and_lookup():
    idx = ExactIndex()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    idx.insert(frag, {"node-a"})
    result = idx.lookup("h1")
    assert result.fragment == frag
    assert result.locations == frozenset({"node-a"})


def test_insert_overwrites():
    idx = ExactIndex()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    idx.insert(frag, {"node-a"})
    idx.insert(frag, {"node-b"})
    entry = idx.lookup("h1")
    assert entry.locations == frozenset({"node-b"})


def test_add_location_idempotent():
    idx = ExactIndex()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    idx.insert(frag, {"node-a"})
    assert idx.add_location("h1", "node-a")
    entry = idx.lookup("h1")
    assert entry.locations == frozenset({"node-a"})


def test_add_location_merges():
    idx = ExactIndex()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("h1", (0.1,), sig, 10, 60.0, 0.5, 1)
    idx.insert(frag, {"node-a"})
    assert idx.add_location("h1", "node-b")
    entry = idx.lookup("h1")
    assert entry.locations == frozenset({"node-a", "node-b"})


def test_lookup_missing():
    idx = ExactIndex()
    assert idx.lookup("missing") is None


def test_add_location_missing():
    idx = ExactIndex()
    assert not idx.add_location("missing", "node-a")
