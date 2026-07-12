"""Tests for InMemoryBackend."""

from membrane.fragment import Fragment
from membrane.persistence.memory_backend import InMemoryBackend
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash: str = "h1", size: int = 100):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 10)
        ),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestInMemoryBackend:
    """Test suite for InMemoryBackend."""

    def setup_method(self):
        self.backend = InMemoryBackend()
        self.backend.flush()

    def test_ping(self):
        assert self.backend.ping() is True

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
