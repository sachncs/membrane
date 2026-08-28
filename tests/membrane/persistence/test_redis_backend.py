"""Tests for redis_backend persistence."""

import pytest

from membrane.fragment import Fragment
from membrane.persistence.redis import Redis
from membrane.signature import Signature

class TestRedisBackend:
    """Test suite for Redis."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.backend = Redis()
        if not self.backend.ping():
            pytest.skip("Redis server not available")
        self.backend.flush()
        yield
        self.backend.flush()

    def test_ping(self):
        assert self.backend.ping()

    def test_store_and_retrieve_fragment(self):
        frag = make_fragment("abc")
        self.backend.store_fragment(frag, "n1", is_primary=True)
        retrieved = self.backend.retrieve_fragment("abc")
        assert retrieved is not None
        assert retrieved.content_hash == "abc"
        assert retrieved.size == 100

    def test_retrieve_missing(self):
        assert self.backend.retrieve_fragment("missing") is None

    def test_delete_fragment(self):
        frag = make_fragment("del")
        self.backend.store_fragment(frag, "n1")
        self.backend.delete_fragment("del")
        assert self.backend.retrieve_fragment("del") is None

    def test_inventory_digest(self):
        self.backend.store_fragment(make_fragment("a", size=10), "n1")
        self.backend.store_fragment(make_fragment("b", size=20), "n1", is_primary=True)
        digest = self.backend.inventory_digest("n1")
        assert digest == {"a": 1, "b": 1}

    def test_list_node_fragments(self):
        self.backend.store_fragment(make_fragment("a"), "n1")
        self.backend.store_fragment(make_fragment("b"), "n1")
        assert self.backend.list_node_fragments("n1") == {"a", "b"}

    def test_record_and_locate(self):
        self.backend.record_location("h1", "n1")
        self.backend.record_location("h1", "n2")
        assert self.backend.locate("h1") == {"n1", "n2"}

    def test_get_primary(self):
        self.backend.store_fragment(make_fragment("p"), "n1", is_primary=True)
        assert self.backend.get_primary("p") == "n1"

    def test_lru_candidates(self):
        self.backend.store_fragment(make_fragment("a"), "n1")
        self.backend.store_fragment(make_fragment("b"), "n1")
        cands = self.backend.lru_candidates(1)
        assert len(cands) == 1
        assert cands[0] in {"a", "b"}

    def test_serialization_roundtrip(self):
        frag = make_fragment("round", size=42)
        data = self.backend.serialize_fragment(frag)
        restored = self.backend.deserialize_fragment(data)
        assert restored.content_hash == frag.content_hash
        assert restored.size == frag.size
        assert restored.embedding == frag.embedding
        assert restored.structural_signature == frag.structural_signature
