"""Resumable transfer over the wire_v3 bidi stream (Phase 3.3.4).

The v3.0.0 wire replaces the v2.0 atomic transfer with a
bidi stream where the client sends :class:`membrane.wire.v3.wire_v3_pb2.ChunkRequest`
messages and the server replies with :class:`membrane.wire.v3.wire_v3_pb2.Chunk`
messages. The :class:`ResumableTransfer` helper threads the
client side; the server side lives in the v3 gRPC transport
that ships in a follow-up release.

A :class:`ResumableReceiver` verifies each chunk on arrival
and surfaces the SHA-256 mismatch as :class:`membrane.errors.CorruptPayloadError`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from membrane.errors import CorruptPayloadError
from membrane.wire.v3.chunks import ChunkManifest, sha256_hex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeCursor:
    """Bookkeeping for a partially-completed transfer.

    Attributes:
        content_hash: Hex digest of the canonical content hash.
        next_index: Index of the chunk the client expects next.
        offset: Offset inside the next chunk for partial reads.
    """

    content_hash: str
    next_index: int = 0
    offset: int = 0


@dataclass
class ResumableTransfer:
    """Client-side chunked / resumable transfer.

    Attributes:
        manifest: The chunk manifest describing the payload.
        cursor: The current cursor (updated as chunks arrive).
        receiver: Optional receiver state for streaming chunks.
    """

    manifest: ChunkManifest
    cursor: ResumeCursor = field(default_factory=lambda: ResumeCursor(""))
    received_chunks: list[bytes] = field(default_factory=list)

    @classmethod
    def new(cls, manifest: ChunkManifest) -> ResumableTransfer:
        """Build a fresh :class:`ResumableTransfer`.

        Args:
            manifest: The chunk manifest describing the payload.

        Returns:
            ResumableTransfer: Initialized with the cursor at the
            manifest's first chunk.
        """
        return cls(
            manifest=manifest,
            cursor=ResumeCursor(
                content_hash=manifest.content_hash,
                next_index=0,
                offset=0,
            ),
        )

    def feed_chunk(self, chunk_index: int, offset: int, data: bytes) -> None:
        """Verify and ingest a single chunk.

        Args:
            chunk_index: Index of the chunk (0-based).
            offset: Offset inside the chunk for partial reads.
            data: Bytes received.

        Raises:
            CorruptPayloadError: When the chunk's SHA-256
                disagrees with the manifest.
        """
        if not self.manifest.verify_chunk(chunk_index, data):
            raise CorruptPayloadError(
                f"chunk {chunk_index} sha256 mismatch (cursor at {self.cursor})"
            )
        if chunk_index >= len(self.received_chunks):
            self.received_chunks.extend([b""] * (chunk_index + 1 - len(self.received_chunks)))
        self.received_chunks[chunk_index] = data
        self.cursor = ResumeCursor(
            content_hash=self.cursor.content_hash,
            next_index=chunk_index + 1,
            offset=offset,
        )

    def all_chunks_received(self) -> bool:
        """Return True once every chunk in the manifest has arrived.

        Returns:
            bool: True when the receiver holds all chunks.
        """
        return len(self.received_chunks) >= len(self.manifest.per_chunk_sha256) and all(
            chunk for chunk in self.received_chunks
        )

    def assemble(self) -> bytes:
        """Concatenate every received chunk into the payload.

        Returns:
            bytes: The full payload.

        Raises:
            CorruptPayloadError: When the receiver is missing one
                or more chunks.
        """
        if not self.all_chunks_received():
            raise CorruptPayloadError(
                f"missing {len(self.manifest.per_chunk_sha256) - len(self.received_chunks)} chunks"
            )
        return b"".join(self.received_chunks)


@dataclass
class ResumableReceiver:
    """Lightweight iterator wrapper around a :class:`ResumableTransfer`.

    Attributes:
        transfer: The :class:`ResumableTransfer` being fed.
    """

    transfer: ResumableTransfer

    def feed(self, chunk_index: int, data: bytes) -> None:
        """Verify and feed one chunk into ``self.transfer``.

        Args:
            chunk_index: Index of the chunk.
            data: Bytes received.
        """
        self.transfer.feed_chunk(chunk_index, 0, data)


@dataclass(frozen=True)
class ResumableProducer:
    """Producer-side state for chunked writes.

    Attributes:
        chunks: The chunk bytes in order.
        manifest: Computed manifest for ``chunks``.
    """

    chunks: tuple[bytes, ...]
    manifest: ChunkManifest

    @classmethod
    def from_payload(
        cls, payload: bytes, chunk_size: int, content_hash: str, compression: int = 0
    ) -> ResumableProducer:
        """Chunk ``payload`` and compute the manifest.

        Args:
            payload: Bytes to chunk.
            chunk_size: Chunk size (must be > 0).
            content_hash: Hex digest of the canonical content hash.
            compression: Compression method id.

        Returns:
            ResumableProducer: A producer with the chunk list and
            the matching manifest.
        """
        manifest = ChunkManifest.from_payload(
            payload=payload,
            content_hash=content_hash,
            chunk_size=chunk_size,
            compression=compression,
        )
        return cls(
            chunks=tuple(manifest.split_payload(payload)),
            manifest=manifest,
        )


def iter_chunks(producer: ResumableProducer) -> Iterator[tuple[int, bytes, str]]:
    """Yield ``(index, bytes, sha256_hex)`` for every chunk.

    Args:
        producer: The producer whose chunks to iterate.

    Yields:
        tuple: ``(index, data, sha256_hex)`` per chunk.
    """
    for index, chunk in enumerate(producer.chunks):
        yield index, chunk, sha256_hex(chunk)


__all__ = [
    "ResumableProducer",
    "ResumableReceiver",
    "ResumableTransfer",
    "ResumeCursor",
    "iter_chunks",
]
