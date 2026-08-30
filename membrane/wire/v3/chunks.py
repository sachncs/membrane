"""Chunk manifest with per-chunk SHA-256 (Phase 3.3.3).

The v3.0.0 wire protocol chunks a payload into fixed-size
bodies and ships a manifest carrying the SHA-256 of every
chunk. The receiving side verifies each chunk on arrival and
rejects the transfer if the manifest hash disagrees with the
recomputed hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def sha256_hex(data: bytes) -> str:
    """Compute the SHA-256 of ``data`` and return its hex digest.

    Args:
        data: Bytes to hash.

    Returns:
        str: 64-character hex digest.
    """
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ChunkManifest:
    """Per-transfer chunk manifest.

    Attributes:
        content_hash: Hex digest of the canonical content hash
            (the receiving side uses it for deduplication).
        chunk_size: Size of every chunk in bytes (the last chunk
            may be shorter).
        total_bytes: Uncompressed payload size in bytes.
        per_chunk_sha256: Tuple of hex digests, one per chunk
            (last entry is the partial chunk's hash when the
            payload does not divide evenly).
        compression: Compression method id (0=raw, 1=zstd, 2=lz4).
    """

    content_hash: str
    chunk_size: int
    total_bytes: int
    per_chunk_sha256: tuple[str, ...]
    compression: int = 0

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        content_hash: str,
        chunk_size: int,
        compression: int = 0,
    ) -> ChunkManifest:
        """Build a manifest by splitting ``payload`` and hashing each chunk.

        Args:
            payload: Bytes to chunk.
            content_hash: Hex digest of the canonical
                :class:`membrane.serialization.PayloadIdentity`
                round-trip.
            chunk_size: Chunk size in bytes (``> 0``).
            compression: Compression method id.

        Returns:
            ChunkManifest: A manifest whose
            :attr:`per_chunk_sha256` length matches the chunk
            count.

        Raises:
            ValueError: When ``chunk_size <= 0``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        digests: list[str] = []
        for offset in range(0, len(payload), chunk_size):
            chunk = payload[offset : offset + chunk_size]
            digests.append(sha256_hex(chunk))
        return cls(
            content_hash=content_hash,
            chunk_size=chunk_size,
            total_bytes=len(payload),
            per_chunk_sha256=tuple(digests),
            compression=compression,
        )

    def split_payload(self, payload: bytes) -> list[bytes]:
        """Split ``payload`` into the manifest's chunk boundaries.

        Args:
            payload: Bytes to split.

        Returns:
            list[bytes]: One entry per chunk.

        Raises:
            ValueError: When ``payload`` length does not match the
            manifest's :attr:`total_bytes`.
        """
        if len(payload) != self.total_bytes:
            raise ValueError(
                f"payload length {len(payload)} does not match manifest total {self.total_bytes}"
            )
        chunks: list[bytes] = []
        for offset in range(0, len(payload), self.chunk_size):
            chunks.append(bytes(payload[offset : offset + self.chunk_size]))
        return chunks

    def verify_chunk(self, index: int, data: bytes) -> bool:
        """Return True when ``data`` matches the manifest's chunk hash.

        Args:
            index: Chunk index (0-based).
            data: Candidate bytes.

        Returns:
            bool: True when the candidate matches.
        """
        if not 0 <= index < len(self.per_chunk_sha256):
            return False
        return self.per_chunk_sha256[index] == sha256_hex(data)


__all__ = ["ChunkManifest", "sha256_hex"]
