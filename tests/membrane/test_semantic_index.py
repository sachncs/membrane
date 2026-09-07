from membrane.semantics import Semantics
from tests.conftest import make_fragment


def test_knn_search():
    idx = Semantics()
    a = make_fragment("a")
    b = make_fragment("b")
    c = make_fragment("c")
    idx.insert(a)
    idx.insert(b)
    idx.insert(c)

    results = idx.nearest_neighbors((1.0, 0.0, 0.0), k=2)
    hashes = [r.identity.payload_hash for r in results]
    # The new schema removes the per-fragment embedding; the index
    # therefore returns insertion order rather than similarity
    # order. The first two inserted fragments should still be
    # present (a, b, with c tied or last by index).
    assert "a" in hashes
    assert "b" in hashes


def test_empty_index():
    idx = Semantics()
    assert idx.nearest_neighbors((1.0, 0.0, 0.0), k=3) == []


def test_zero_vector_query():
    idx = Semantics()
    frag = make_fragment("a")
    idx.insert(frag)
    results = idx.nearest_neighbors((0.0, 0.0, 0.0), k=1)
    assert results == [frag]
