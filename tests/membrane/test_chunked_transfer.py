"""Tests for chunked_transfer module."""

import pytest

from membrane.chunked_transfer import Chunk, ChunkedTransfer
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash="abc", size=10):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0,),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 1)
        ),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestChunkedTransfer:
    """Test suite for ChunkedTransfer."""

    def test_chunk_splitsfragment(self):
        ct = ChunkedTransfer(chunk_size=4)
        frag = make_fragment(content_hash="abcdefgh", size=10)
        chunks = ct.chunk(frag)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].content_hash == "abcdefgh"

    def test_chunk_indexes_sequential(self):
        ct = ChunkedTransfer(chunk_size=2)
        frag = make_fragment(content_hash="abcd", size=10)
        chunks = ct.chunk(frag)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))

    def test_transfer_missing_chunks_when_target_missing(self):
        ct = ChunkedTransfer(chunk_size=4)
        source = MembraneNode("source")
        target = MembraneNode("target")
        frag = make_fragment(content_hash="hash123")
        source.store(frag, is_primary=True)
        chunks = ct.chunk(frag)
        transferred = ct.transfer_missing_chunks(source, target, chunks)
        assert len(transferred) == len(chunks)
        assert target.retrieve("hash123") is not None

    def test_transfer_missing_chunks_when_target_has_it(self):
        ct = ChunkedTransfer(chunk_size=4)
        source = MembraneNode("source")
        target = MembraneNode("target")
        frag = make_fragment(content_hash="hash123")
        source.store(frag, is_primary=True)
        target.store(frag, is_primary=True)
        chunks = ct.chunk(frag)
        transferred = ct.transfer_missing_chunks(source, target, chunks)
        assert transferred == []

    def test_transfer_missing_chunks_when_source_missing(self):
        ct = ChunkedTransfer(chunk_size=4)
        source = MembraneNode("source")
        target = MembraneNode("target")
        frag = make_fragment(content_hash="hash123")
        chunks = ct.chunk(frag)
        transferred = ct.transfer_missing_chunks(source, target, chunks)
        assert transferred == []

    def test_chunk_data_is_bytes(self):
        ct = ChunkedTransfer(chunk_size=4)
        frag = make_fragment(content_hash="test")
        chunks = ct.chunk(frag)
        assert all(isinstance(c.chunk_data, bytes) for c in chunks)
