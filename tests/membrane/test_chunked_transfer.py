from tests.conftest import make_fragment

"""Tests for chunked_transfer module."""

import pytest

from membrane.chunks import Chunk, Chunks
from membrane.fragment import Fragment
from membrane.node import Node


class TestChunkedTransfer:
    """Test suite for Chunks."""

    def test_chunk_splitsfragment(self):
        ct = Chunks(chunk_size=4)
        frag = make_fragment(content_hash="abcdefgh", size=10)
        chunks = ct.chunk(frag)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].content_hash == "abcdefgh"

    def test_chunk_indexes_sequential(self):
        ct = Chunks(chunk_size=2)
        frag = make_fragment(content_hash="abcd", size=10)
        chunks = ct.chunk(frag)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))

    def test_transfer_missing_chunks_when_target_missing(self):
        ct = Chunks(chunk_size=4)
        source = Node("source")
        target = Node("target")
        frag = make_fragment(content_hash="hash123")
        source.store(frag, is_primary=True)
        chunks = ct.chunk(frag)
        transferred = ct.transfer_missing_chunks(source, target, chunks)
        assert len(transferred) == len(chunks)
        assert target.retrieve("hash123") is not None

    def test_transfer_missing_chunks_when_target_has_it(self):
        ct = Chunks(chunk_size=4)
        source = Node("source")
        target = Node("target")
        frag = make_fragment(content_hash="hash123")
        source.store(frag, is_primary=True)
        target.store(frag, is_primary=True)
        chunks = ct.chunk(frag)
        transferred = ct.transfer_missing_chunks(source, target, chunks)
        assert transferred == []

    def test_transfer_missing_chunks_when_source_missing(self):
        ct = Chunks(chunk_size=4)
        source = Node("source")
        target = Node("target")
        frag = make_fragment(content_hash="hash123")
        chunks = ct.chunk(frag)
        transferred = ct.transfer_missing_chunks(source, target, chunks)
        assert transferred == []

    def test_chunk_data_is_bytes(self):
        ct = Chunks(chunk_size=4)
        frag = make_fragment(content_hash="test")
        chunks = ct.chunk(frag)
        assert all(isinstance(c.chunk_data, bytes) for c in chunks)
