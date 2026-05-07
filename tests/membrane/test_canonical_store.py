"""Tests for canonical_store module."""

import pytest

from membrane.canonical_store import CanonicalRef, CanonicalStore
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0,),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 1)
        ),
        size=10,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestCanonicalStore:
    """Test suite for CanonicalStore."""

    def test_store_canonical_new(self):
        cs = CanonicalStore()
        frag = make_fragment("h1")
        ref = cs.store_canonical(frag, "t1")
        assert ref.canonical_hash == "h1"
        assert "t1" in ref.tenant_ids

    def test_store_canonical_deduplicates(self):
        cs = CanonicalStore()
        frag = make_fragment("h1")
        cs.store_canonical(frag, "t1")
        cs.store_canonical(frag, "t2")
        ref = cs.store_canonical(frag, "t3")
        assert ref.tenant_ids == frozenset({"t1", "t2", "t3"})
        assert len(cs.canonical_fragments) == 1

    def test_retrieve_canonical(self):
        cs = CanonicalStore()
        frag = make_fragment("h1")
        ref = cs.store_canonical(frag, "t1")
        retrieved = cs.retrieve_canonical(ref)
        assert retrieved == frag

    def test_retrieve_canonical_missing(self):
        cs = CanonicalStore()
        ref = CanonicalRef("missing", frozenset())
        assert cs.retrieve_canonical(ref) is None

    def test_get_shared_fragments(self):
        cs = CanonicalStore()
        f1 = make_fragment("h1")
        f2 = make_fragment("h2")
        cs.store_canonical(f1, "t1")
        cs.store_canonical(f1, "t2")
        cs.store_canonical(f2, "t2")
        shared = cs.get_shared_fragments("t1")
        assert len(shared) == 1
        assert shared[0].content_hash == "h1"

    def test_get_shared_fragments_no_match(self):
        cs = CanonicalStore()
        f1 = make_fragment("h1")
        cs.store_canonical(f1, "t1")
        assert cs.get_shared_fragments("t2") == []

    def test_lru_eviction_on_overflow(self):
        cs = CanonicalStore(max_entries=2)
        cs.store_canonical(make_fragment("a"), "t1")
        cs.store_canonical(make_fragment("b"), "t1")
        cs.store_canonical(make_fragment("c"), "t1")
        assert len(cs.canonical_fragments) == 2
        assert "a" not in cs.canonical_fragments

    def test_lru_keeps_recently_accessed(self):
        cs = CanonicalStore(max_entries=2)
        cs.store_canonical(make_fragment("a"), "t1")
        cs.store_canonical(make_fragment("b"), "t1")
        # Access "a" to make it recently used
        cs.retrieve_canonical(CanonicalRef("a", frozenset()))
        cs.store_canonical(make_fragment("c"), "t1")
        assert "a" in cs.canonical_fragments
        assert "b" not in cs.canonical_fragments
