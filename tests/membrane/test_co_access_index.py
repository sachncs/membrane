from membrane.co_access_index import CoAccessIndex


def test_record_and_lookup():
    idx = CoAccessIndex()
    idx.record_access("a", "b")
    idx.record_access("a", "c")
    neighbors = idx.lookup("a")
    assert neighbors == {"b", "c"}


def test_record_batch():
    idx = CoAccessIndex()
    idx.record_batch(["a", "b", "c"])
    assert idx.lookup("a") == {"b", "c"}
    assert idx.lookup("b") == {"a", "c"}


def test_self_access_ignored():
    idx = CoAccessIndex()
    idx.record_access("a", "a")
    assert idx.lookup("a") == set()


def test_lookup_missing():
    idx = CoAccessIndex()
    assert idx.lookup("z") == set()
