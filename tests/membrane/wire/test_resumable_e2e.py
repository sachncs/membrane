"""End-to-end test for the wire_v3 chunked + resumable flow (Phase 3.3.3-3.3.4 follow-up)."""

from __future__ import annotations

import pytest

from membrane.wire.v3 import (
    ChunkManifest,
    ResumableProducer,
    ResumableReceiver,
    ResumableTransfer,
    sha256_hex,
)


class TestChunkedManifest:
    def test_from_payload_chunk_size_divides(self):
        payload = b"a" * 4096
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=1024)
        assert len(manifest.per_chunk_sha256) == 4
        for digest in manifest.per_chunk_sha256:
            assert len(digest) == 64


class TestResumableFlow:
    def test_send_all_chunks_consumer_receives_all(self):
        payload = b"abcdefghij"
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=2)
        producer = ResumableProducer.from_payload(
            payload=payload, chunk_size=2, content_hash="h" * 64
        )
        transfer = ResumableTransfer.new(manifest)
        receiver = ResumableReceiver(transfer=transfer)
        # Manually walk the producer; the receiver feeds each chunk.
        for index, chunk in enumerate(producer.chunks):
            assert transfer.manifest.verify_chunk(index, chunk)
            transfer.feed_chunk(index, 0, chunk)
        assert transfer.all_chunks_received()
        assert transfer.assemble() == payload
        # The receiver wraps the same transfer; sanity check the
        # facade.
        assert receiver.transfer is transfer

    def test_resume_after_partial_delivery(self):
        """A consumer that misses chunk 1 and re-requests it can resume."""
        payload = b"x" * 1024
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=256)
        transfer = ResumableTransfer.new(manifest)
        # Consumer received chunks 0, 2, 3 but missed 1.
        chunks = manifest.split_payload(payload)
        for index in (0, 2, 3):
            transfer.feed_chunk(index, 0, chunks[index])
        assert not transfer.all_chunks_received()
        # Resume: re-deliver chunk 1.
        transfer.feed_chunk(1, 0, chunks[1])
        assert transfer.all_chunks_received()
        assert transfer.assemble() == payload

    def test_duplicate_chunk_does_not_advance(self):
        """A duplicate chunk is dropped, not double-counted."""
        manifest = ChunkManifest.from_payload(b"abcd", "h" * 64, chunk_size=2)
        transfer = ResumableTransfer.new(manifest)
        transfer.feed_chunk(0, 0, b"ab")
        transfer.feed_chunk(0, 0, b"ab")  # duplicate
        # The cursor advanced to index 1, not 2.
        assert transfer.cursor.next_index == 1
        transfer.feed_chunk(1, 0, b"cd")
        assert transfer.all_chunks_received()

    def test_tampered_chunk_raises_corrupt(self):
        from membrane.errors import CorruptPayloadError

        manifest = ChunkManifest.from_payload(b"abcd", "h" * 64, chunk_size=2)
        transfer = ResumableTransfer.new(manifest)
        with pytest.raises(CorruptPayloadError):
            transfer.feed_chunk(0, 0, b"XX")  # wrong chunk 0

    def test_split_payload_round_trip(self):
        payload = b"hello" * 50
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=128)
        chunks = manifest.split_payload(payload)
        assert b"".join(chunks) == payload

    def test_manifest_size_zero(self):
        manifest = ChunkManifest.from_payload(b"", "h" * 64, chunk_size=2)
        assert manifest.per_chunk_sha256 == ()
        assert manifest.total_bytes == 0

    def test_manifest_single_chunk(self):
        manifest = ChunkManifest.from_payload(b"abc", "h" * 64, chunk_size=2)
        # payload_size=3, chunk_size=2 -> 2 chunks
        assert len(manifest.per_chunk_sha256) == 2
