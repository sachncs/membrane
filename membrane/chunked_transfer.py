"""ChunkedTransfer: split fragments into chunks for partial retrieval.

This module implements a minimal chunked-transfer protocol used when
moving fragments between :class:`~membrane.membrane_node.MembraneNode`
instances. Large payloads can be split into fixed-size chunks so that
partial progress is preserved across connection failures and so that
remote peers only need to fetch the chunks they are missing.

Note:
    This implementation chunks the fragment's *content hash string*
    rather than its underlying tensor bytes — it is therefore intended
    as a lightweight *transport sharding* mechanism used by the
    :class:`~membrane.transfer_service.TransferService` and tests,
    not as a byte-stream chunker for raw KV tensors. Real tensor
    transport goes through :class:`~membrane.transport.http_server
    .HTTPServer` / :class:`~membrane.transport.grpc_server.GrpcServer`
    which stream the actual payload.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode


@dataclass(frozen=True)
class Chunk:
    """A slice of a fragment for incremental transport.

    Chunks are identified by their parent fragment's ``content_hash``
    and their position within the chunk list. Together with the
    parent fragment, this is sufficient to reassemble the chunk
    sequence on the receiving side.

    Attributes:
        content_hash: Content hash of the parent fragment. Acts as
            the routing key on both source and target nodes.
        chunk_index: Zero-based position of this chunk within the
            parent's chunk list. Combined with ``chunk_size`` this
            uniquely identifies the slice.
        chunk_data: Raw payload bytes for this slice. The current
            implementation stores a UTF-8 encoded portion of the
            parent's content hash.
    """

    content_hash: str
    chunk_index: int
    chunk_data: bytes


class ChunkedTransfer:
    """Splits fragments into chunks and transfers missing chunks incrementally.

    The class is stateless beyond its configured ``chunk_size``. It
    acts as a thin coordinator between two
    :class:`~membrane.membrane_node.MembraneNode` instances, deciding
    which chunks still need to flow across the network.

    Attributes:
        chunk_size: Number of bytes per chunk. Must be positive; the
            default ``64`` keeps each chunk small enough for
            connection-oriented transports without excessive framing
            overhead.
    """

    def __init__(self, chunk_size: int = 64) -> None:
        """Initialize the chunker with the desired chunk size.

        Args:
            chunk_size: Number of bytes per chunk. Smaller values
                improve granularity at the cost of more transfer
                round-trips.
        """
        self.chunk_size = chunk_size

    def chunk(self, fragment: Fragment) -> list[Chunk]:
        """Split a fragment into fixed-size chunks.

        The payload used for chunking is the UTF-8 encoding of the
        fragment's ``content_hash``. This is sufficient for
        content-addressed transport (the receiving node resolves the
        hash back to the full fragment) while keeping the chunker
        independent of the fragment's underlying tensor type.

        Args:
            fragment: The fragment to chunk. Only its
                ``content_hash`` is read.

        Returns:
            list[Chunk]: Ordered list of chunks with strictly
            increasing ``chunk_index`` values, suitable for
            streaming or reassembly.
        """
        data = fragment.content_hash.encode("utf-8")
        chunks: list[Chunk] = []
        # Walk the payload in fixed-size windows; integer division
        # gives the zero-based chunk index for each window.
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

        The transfer is content-driven: if the source does not have
        the parent fragment, or the target already has it, no chunks
        are transferred. Otherwise, the parent fragment is copied to
        the target as a non-primary replica and the original chunk
        list is returned.

        Args:
            source: Node holding the parent fragment. Its
                :meth:`~membrane.membrane_node.MembraneNode.retrieve`
                method must return the parent for the transfer to
                proceed.
            target: Node that should receive missing chunks. It must
                expose :meth:`~membrane.membrane_node.MembraneNode
                .store` and
                :meth:`~membrane.membrane_node.MembraneNode.retrieve`.
            chunks: The full chunk list for the fragment. Only the
                first chunk is inspected to recover the parent
                ``content_hash``.

        Returns:
            list[Chunk]: The chunks that were transferred. In the
            current implementation this is either the full input
            ``chunks`` list (when a real transfer occurred) or an
            empty list (when the source lacked the fragment or the
            target already had it).
        """
        if not chunks:
            return []

        # Early-out: if the source cannot produce the parent
        # fragment there is nothing to transfer. The first probe
        # also implicitly validates that the chunk list is
        # consistent with the source's content.
        if source.retrieve(chunks[0].content_hash) is None:
            return []

        parent = source.retrieve(chunks[0].content_hash)
        if parent is None:
            # Double-checked lookup to satisfy type-narrowing tools
            # and to defend against races in concurrent transfers.
            return []

        if target.retrieve(parent.content_hash) is not None:
            # Target already has the parent fragment — no need to
            # transfer any chunks. Idempotent and safe to call
            # repeatedly.
            return []

        # Store as a non-primary replica so that the target can
        # serve reads but won't act as the canonical owner for the
        # fragment.
        target.store(parent, is_primary=False)
        return chunks
