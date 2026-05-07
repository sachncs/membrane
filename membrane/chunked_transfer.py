"""ChunkedTransfer: split fragments into chunks for partial retrieval."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode


@dataclass(frozen=True)
class Chunk:
    """A slice of a fragment for incremental transport.

    Attributes:
        content_hash: Parent fragment hash.
        chunk_index: Position within the fragment.
        chunk_data: Raw bytes payload.
    """

    content_hash: str
    chunk_index: int
    chunk_data: bytes


class ChunkedTransfer:
    """Splits fragments into chunks and transfers missing chunks incrementally."""

    def __init__(self, chunk_size: int = 64) -> None:
        """Initialize with chunk size.

        Args:
            chunk_size: Number of bytes per chunk.
        """
        """Initialize with chunk size.

        Args:
            chunk_size: Number of bytes per chunk.
        """
        self.chunk_size = chunk_size

    def chunk(self, fragment: Fragment) -> list[Chunk]:
        """Split a fragment into fixed-size chunks.

        Args:
            fragment: Fragment to chunk.

        Returns:
            Ordered list of chunks.
        """
        data = fragment.content_hash.encode("utf-8")
        chunks: list[Chunk] = []
        for i in range(0, len(data), self.chunk_size):
            chunk_data = data[i : i + self.chunk_size]
            chunks.append(
                Chunk(
                    content_hash=fragment.content_hash,
                    chunk_index=i // self.chunk_size,
                    chunk_data=chunk_data,
                )
            )
        return chunks

    def transfer_missing_chunks(
        self,
        source: MembraneNode,
        target: MembraneNode,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Transfer chunks that are missing on the target node.

        Args:
            source: Node holding the parent fragment.
            target: Node to receive missing chunks.
            chunks: Full chunk list for the fragment.

        Returns:
            List of chunks that were actually transferred.
        """
        if source.retrieve(chunks[0].content_hash) is None:
            return []

        parent = source.retrieve(chunks[0].content_hash)
        if parent is None:
            return []

        if target.retrieve(parent.content_hash) is not None:
            return []

        target.store(parent, is_primary=False)
        return chunks
