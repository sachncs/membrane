from membrane.exacts import Exacts
from tests.conftest import make_fragment


def test_index_and_lookup():
    idx = Exacts()
    frag = make_fragment("h1")
    idx.insert(frag, {"node-a"})
    result = idx.lookup("h1")
    assert result.fragment == frag
    assert result.locations == frozenset({"node-a"})


def test_insert_overwrites():
    idx = Exacts()
    frag = make_fragment("h1")
    idx.insert(frag, {"node-a"})
    idx.insert(frag, {"node-b"})
    entry = idx.lookup("h1")
    assert entry.locations == frozenset({"node-b"})


def test_add_location_idempotent():
    idx = Exacts()
    frag = make_fragment("h1")
    idx.insert(frag, {"node-a"})
    assert idx.add_location("h1", "node-a")
    entry = idx.lookup("h1")
    assert entry.locations == frozenset({"node-a"})


def test_add_location_merges():
    idx = Exacts()
    frag = make_fragment("h1")
    idx.insert(frag, {"node-a"})
    assert idx.add_location("h1", "node-b")
    entry = idx.lookup("h1")
    assert entry.locations == frozenset({"node-a", "node-b"})


def test_lookup_missing():
    idx = Exacts()
    assert idx.lookup("missing") is None


def test_add_location_missing():
    idx = Exacts()
    assert not idx.add_location("missing", "node-a")
