"""Tests for _PositionalIndex using an interval tree."""

import pytest

from membrane.fragment import Fragment
from membrane.positions import _PositionalIndex
from membrane.signature import Signature

def test_overlap_query():
    idx = _PositionalIndex()
    frag = make_fragment("h1", (100, 200))
    idx.insert(frag)
    results = idx.find_overlapping(150, 250)
    assert len(results) == 1
    assert results[0].content_hash == "h1"

def test_adjacent_query():
    idx = _PositionalIndex()
    frag = make_fragment("h1", (100, 200))
    idx.insert(frag)
    results = idx.find_adjacent(200, max_gap=10)
    assert len(results) == 1

def test_no_overlap():
    idx = _PositionalIndex()
    frag = make_fragment("h1", (100, 200))
    idx.insert(frag)
    assert idx.find_overlapping(300, 400) == []

# --- Tests designed to break the interval tree algorithm ---

def test_remove_existingfragment():
    idx = _PositionalIndex()
    frag = make_fragment("h1", (100, 200))
    idx.insert(frag)
    assert idx.remove("h1") is True
    assert idx.find_overlapping(100, 200) == []

def test_remove_nonexistentfragment():
    idx = _PositionalIndex()
    assert idx.remove("missing") is False

def test_many_intervals_stress():
    """Insert many intervals and verify all are retrievable."""
    idx = _PositionalIndex()
    n = 200
    for i in range(n):
        idx.insert(make_fragment(f"h{i}", (i * 10, i * 10 + 5)))
    # Query that should overlap many intervals
    results = idx.find_overlapping(0, n * 10)
    assert len(results) == n

def test_nested_intervals():
    """Nested intervals should all be found."""
    idx = _PositionalIndex()
    idx.insert(make_fragment("outer", (0, 100)))
    idx.insert(make_fragment("inner", (40, 60)))
    results = idx.find_overlapping(45, 55)
    hashes = {f.content_hash for f in results}
    assert hashes == {"outer", "inner"}

def test_same_start_different_ends():
    """Multiple intervals with same start but different ends."""
    idx = _PositionalIndex()
    idx.insert(make_fragment("a", (50, 60)))
    idx.insert(make_fragment("b", (50, 70)))
    idx.insert(make_fragment("c", (50, 80)))
    results = idx.find_overlapping(55, 65)
    hashes = {f.content_hash for f in results}
    assert hashes == {"a", "b", "c"}

def test_boundary_overlap_exact():
    """Interval [10, 20] should overlap with [20, 30] at point 20."""
    idx = _PositionalIndex()
    idx.insert(make_fragment("a", (10, 20)))
    results = idx.find_overlapping(20, 30)
    assert len(results) == 1

def test_boundary_no_overlap_off_by_one():
    """Interval [10, 20] should NOT overlap with [21, 30]."""
    idx = _PositionalIndex()
    idx.insert(make_fragment("a", (10, 20)))
    assert idx.find_overlapping(21, 30) == []

def test_adjacent_gap_zero_exact_touch():
    """max_gap=0 means fragments must touch exactly."""
    idx = _PositionalIndex()
    idx.insert(make_fragment("a", (10, 20)))
    results = idx.find_adjacent(20, max_gap=0)
    assert len(results) == 1

def test_adjacent_gap_one_near_touch():
    """max_gap=1 should include fragments 1 token away."""
    idx = _PositionalIndex()
    idx.insert(make_fragment("a", (10, 20)))
    results = idx.find_adjacent(22, max_gap=1)
    assert len(results) == 0
    results2 = idx.find_adjacent(21, max_gap=1)
    assert len(results2) == 1

def test_tree_balance_after_many_insertions():
    """After many insertions in sorted order, tree should still be balanced."""
    idx = _PositionalIndex()
    for i in range(100):
        idx.insert(make_fragment(f"h{i}", (i, i + 1)))
    # All should be findable
    results = idx.find_overlapping(0, 99)
    assert len(results) == 100

def test_remove_and_reinsert():
    """Remove then reinsert should work correctly."""
    idx = _PositionalIndex()
    frag = make_fragment("a", (10, 20))
    idx.insert(frag)
    idx.remove("a")
    idx.insert(frag)
    results = idx.find_overlapping(10, 20)
    assert len(results) == 1

def test_complex_scenario_multiple_operations():
    """Mix of inserts, removes, and queries."""
    idx = _PositionalIndex()
    for i in range(50):
        idx.insert(make_fragment(f"h{i}", (i * 2, i * 2 + 3)))
    # Remove every other
    for i in range(0, 50, 2):
        idx.remove(f"h{i}")
    results = idx.find_overlapping(0, 100)
    hashes = {f.content_hash for f in results}
    # Only odd-indexed fragments remain
    expected = {f"h{i}" for i in range(1, 50, 2)}
    assert hashes == expected
