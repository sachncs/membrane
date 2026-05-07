"""Tests for memory_object module."""

import pytest

from membrane.fragment import Fragment
from membrane.memory_object import MemoryObject
from membrane.structural_signature import StructuralSignature


class MockMemoryObject:
    """Concrete implementation of MemoryObject for testing."""

    def __init__(
        self,
        content_hash: str,
        semantic_hash: str,
        size_bytes: int,
        token_count: int,
        reuse_score: float,
    ) -> None:
        self.content_hash = content_hash
        self.semantic_hash = semantic_hash
        self.size_bytes = size_bytes
        self.token_count = token_count
        self.reuse_score = reuse_score

    def materialize(self) -> Fragment:
        return Fragment(
            content_hash=self.content_hash,
            embedding=(0.0,),
            structural_signature=StructuralSignature(
                model_id="mock", layer_range=(0, 0), token_span=(0, 0)
            ),
            size=self.size_bytes,
            ttl=3600.0,
            reuse_score=self.reuse_score,
            version_id=1,
        )

    @classmethod
    def from_fragment(cls, fragment: Fragment) -> "MockMemoryObject":
        return cls(
            content_hash=fragment.content_hash,
            semantic_hash="semantic-" + fragment.content_hash,
            size_bytes=fragment.size,
            token_count=0,
            reuse_score=fragment.reuse_score,
        )


class TestMemoryObject:
    """Test suite for MemoryObject protocol."""

    def test_isinstance_protocol(self):
        obj = MockMemoryObject("h1", "sh1", 100, 10, 0.5)
        assert isinstance(obj, MemoryObject)

    def test_materialize_returns_fragment(self):
        obj = MockMemoryObject("h1", "sh1", 100, 10, 0.5)
        frag = obj.materialize()
        assert isinstance(frag, Fragment)
        assert frag.content_hash == "h1"
        assert frag.size == 100
        assert frag.reuse_score == 0.5

    def test_from_fragment_reconstructs(self):
        frag = Fragment(
            content_hash="h2",
            embedding=(0.0,),
            structural_signature=StructuralSignature(
                model_id="m", layer_range=(0, 1), token_span=(0, 1)
            ),
            size=50,
            ttl=3600.0,
            reuse_score=0.8,
            version_id=1,
        )
        obj = MockMemoryObject.from_fragment(frag)
        assert obj.content_hash == "h2"
        assert obj.reuse_score == 0.8
        assert obj.size_bytes == 50

    def test_protocol_attributes(self):
        obj = MockMemoryObject("h", "sh", 10, 5, 0.3)
        assert obj.content_hash == "h"
        assert obj.semantic_hash == "sh"
        assert obj.token_count == 5

    def test_non_compliant_object_fails_isinstance(self):
        class Bad:
            pass

        assert not isinstance(Bad(), MemoryObject)
