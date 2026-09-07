"""Tests for the wire_v3 chunked / resumable transport (Phase 3.3.3 + 3.3.4)."""

from __future__ import annotations

import pytest

from membrane.errors import CorruptPayloadError
from membrane.wire.v3 import (
    ChunkManifest,
    ResumableProducer,
    ResumableReceiver,
    ResumableTransfer,
)
from membrane.wire.v3.chunks import sha256_hex


class TestSha256Hex:
    def test_returns_64_char_hex(self):
        digest = sha256_hex(b"hello")
        assert len(digest) == 64
        assert int(digest, 16) >= 0


class TestChunkManifest:
    def test_chunk_count(self):
        manifest = ChunkManifest.from_payload(b"abcdefghij", "h" * 64, chunk_size=4)
        assert len(manifest.per_chunk_sha256) == 3

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError):
            ChunkManifest.from_payload(b"x", "h" * 64, chunk_size=0)

    def test_split_and_round_trip(self):
        payload = b"abcdefghij"
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=4)
        chunks = manifest.split_payload(payload)
        assert b"".join(chunks) == payload

    def test_split_wrong_length_raises(self):
        manifest = ChunkManifest.from_payload(b"abcd", "h" * 64, chunk_size=2)
        with pytest.raises(ValueError):
            manifest.split_payload(b"abc")

    def test_verify_chunk(self):
        payload = b"abcde"
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=2)
        assert manifest.verify_chunk(0, b"ab") is True
        assert manifest.verify_chunk(1, b"cd") is True
        assert manifest.verify_chunk(2, b"e") is True
        assert manifest.verify_chunk(0, b"XX") is False

    def test_verify_chunk_out_of_range_returns_false(self):
        manifest = ChunkManifest.from_payload(b"ab", "h" * 64, chunk_size=2)
        assert manifest.verify_chunk(99, b"x") is False


class TestResumableTransfer:
    def test_feeds_chunks_in_order(self):
        payload = b"abcdefghij"
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=4)
        transfer = ResumableTransfer.new(manifest)
        for idx, chunk in enumerate(manifest.split_payload(payload)):
            transfer.feed_chunk(idx, 0, chunk)
        assert transfer.all_chunks_received()
        assert transfer.assemble() == payload

    def test_missing_chunk_raises(self):
        manifest = ChunkManifest.from_payload(b"abcdef", "h" * 64, chunk_size=2)
        transfer = ResumableTransfer.new(manifest)
        transfer.feed_chunk(0, 0, b"ab")
        # Skip chunk 1; chunk 2 arrives
        transfer.feed_chunk(2, 0, b"ef")
        assert not transfer.all_chunks_received()
        with pytest.raises(CorruptPayloadError):
            transfer.assemble()

    def test_corrupt_chunk_raises(self):
        manifest = ChunkManifest.from_payload(b"abc", "h" * 64, chunk_size=1)
        transfer = ResumableTransfer.new(manifest)
        with pytest.raises(CorruptPayloadError):
            transfer.feed_chunk(0, 0, b"X")


class TestResumableReceiver:
    def test_feeds_into_transfer(self):
        payload = b"abc"
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=1)
        receiver = ResumableReceiver(transfer=ResumableTransfer.new(manifest))
        for idx, chunk in enumerate(manifest.split_payload(payload)):
            receiver.feed(idx, chunk)
        assert receiver.transfer.all_chunks_received()


class TestResumableProducer:
    def test_iter_chunks_yields_each_chunk(self):
        payload = b"abcdef"
        producer = ResumableProducer.from_payload(
            payload=payload, chunk_size=2, content_hash="h" * 64
        )
        chunks = list(producer.chunks)
        assert b"".join(chunks) == payload
        assert len(chunks) == 3
