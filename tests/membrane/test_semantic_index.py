from membrane.fragment import Fragment
from membrane.semantic_index import SemanticIndex
from membrane.structural_signature import StructuralSignature


def test_knn_search():
    idx = SemanticIndex()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    a = Fragment("a", (1.0, 0.0, 0.0), sig, 10, 60.0, 0.5, 1)
    b = Fragment("b", (0.0, 1.0, 0.0), sig, 10, 60.0, 0.5, 1)
    c = Fragment("c", (0.9, 0.1, 0.0), sig, 10, 60.0, 0.5, 1)
    idx.insert(a)
    idx.insert(b)
    idx.insert(c)

    results = idx.nearest_neighbors((1.0, 0.0, 0.0), k=2)
    hashes = [r.content_hash for r in results]
    assert "a" in hashes
    assert "c" in hashes
    assert "b" not in hashes


def test_empty_index():
    idx = SemanticIndex()
    assert idx.nearest_neighbors((1.0, 0.0, 0.0), k=3) == []


def test_zero_vector_query():
    idx = SemanticIndex()
    sig = StructuralSignature("m", (0, 1), (0, 10))
    frag = Fragment("a", (1.0, 0.0, 0.0), sig, 10, 60.0, 0.5, 1)
    idx.insert(frag)
    results = idx.nearest_neighbors((0.0, 0.0, 0.0), k=1)
    assert results == [frag]
