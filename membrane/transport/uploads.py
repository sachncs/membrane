"""Resumable chunked uploads for large blobs.

A sender posts a manifest (chunk size, total size, SHA-256 of every chunk
and of the whole payload), then puts the chunks in any order. The node
answers each step with the chunks it already holds, so after a dropped
connection only the missing chunks are sent again. When the last chunk
arrives the payload is assembled, verified against the whole-payload
digest, and stored.

Staged uploads are bounded (count and bytes) and dropped after
:data:`UPLOAD_TTL_SEC` without progress.
"""

import threading
import time
from dataclasses import dataclass, field

from membrane.errors import CorruptPayloadError
from membrane.wire.v3.chunks import ChunkManifest, sha256_hex
from membrane.wire.v3.resumable import ResumableTransfer

UPLOAD_TTL_SEC = 3600.0
MAX_UPLOADS = 64
MAX_STAGED_BYTES = 1 << 30
MAX_CHUNK_BYTES = 8 << 20


class UploadError(ValueError):
    """The upload request is invalid or exceeds a limit."""


@dataclass
class Upload:
    """One staged upload.

    Attributes:
        transfer: Received chunks, verified against the manifest.
        sha256: Digest of the whole payload.
        touched: Monotonic time of the last progress.
    """

    transfer: ResumableTransfer
    sha256: str
    touched: float = field(default_factory=time.monotonic)

    @property
    def received(self) -> list[int]:
        """Indices of the chunks already held."""
        return [i for i, chunk in enumerate(self.transfer.received_chunks) if chunk]


class UploadRegistry:
    """Staged uploads keyed by payload reference."""

    def __init__(self, max_bytes: int, max_uploads: int = MAX_UPLOADS, max_staged: int = MAX_STAGED_BYTES) -> None:
        """Create an empty registry.

        Args:
            max_bytes: Largest payload accepted.
            max_uploads: Concurrent uploads allowed.
            max_staged: Total payload bytes that may be staged at once.
        """
        self.max_bytes = max_bytes
        self.max_uploads = max_uploads
        self.max_staged = max_staged
        self.__uploads: dict[str, Upload] = {}
        self.__lock = threading.Lock()

    def begin(self, payload_ref: str, chunk_size: int, total_bytes: int, chunks: list[str], sha256: str) -> list[int]:
        """Start (or resume) an upload.

        Args:
            payload_ref: Content-store key.
            chunk_size: Size of every chunk but the last.
            total_bytes: Payload size.
            chunks: SHA-256 of each chunk.
            sha256: SHA-256 of the whole payload.

        Returns:
            list[int]: Indices already received (empty for a new upload).

        Raises:
            UploadError: On an inconsistent manifest or exceeded limits.
        """
        expected = -(-total_bytes // chunk_size) if chunk_size > 0 else -1
        if not 0 < chunk_size <= MAX_CHUNK_BYTES or total_bytes > self.max_bytes or len(chunks) != expected:
            raise UploadError("manifest does not describe the payload, or it is too large")
        with self.__lock:
            self.__expire()
            current = self.__uploads.get(payload_ref)
            if current is not None and current.sha256 == sha256:
                current.touched = time.monotonic()
                return current.received
            staged = sum(u.transfer.manifest.total_bytes for u in self.__uploads.values())
            if len(self.__uploads) >= self.max_uploads or staged + total_bytes > self.max_staged:
                raise UploadError("too many uploads in progress; retry later")
            manifest = ChunkManifest(
                content_hash=payload_ref, chunk_size=chunk_size, total_bytes=total_bytes, per_chunk_sha256=tuple(chunks)
            )
            self.__uploads[payload_ref] = Upload(ResumableTransfer.new(manifest), sha256)
            return []

    def add_chunk(self, payload_ref: str, index: int, data: bytes) -> bytes | None:
        """Accept one chunk; return the assembled payload once complete.

        Args:
            payload_ref: Content-store key.
            index: Chunk index.
            data: Chunk bytes.

        Returns:
            bytes | None: The verified payload when this chunk completed it.

        Raises:
            UploadError: For an unknown upload or a chunk that fails its
                digest, or a payload that fails the whole-payload digest.
        """
        with self.__lock:
            upload = self.__uploads.get(payload_ref)
            if upload is None:
                raise UploadError("no upload in progress; post the manifest first")
            try:
                upload.transfer.feed_chunk(index, 0, data)
            except CorruptPayloadError as exc:
                raise UploadError(str(exc)) from exc
            upload.touched = time.monotonic()
            if not upload.transfer.all_chunks_received():
                return None
            del self.__uploads[payload_ref]
        payload = upload.transfer.assemble()
        if sha256_hex(payload) != upload.sha256:
            raise UploadError("assembled payload does not match its digest")
        return payload

    def received(self, payload_ref: str) -> list[int]:
        """Indices received so far for an upload.

        Args:
            payload_ref: Content-store key.

        Returns:
            list[int]: Received chunk indices (empty when unknown).
        """
        with self.__lock:
            upload = self.__uploads.get(payload_ref)
            return upload.received if upload is not None else []

    def __expire(self) -> None:
        """Drop uploads idle longer than :data:`UPLOAD_TTL_SEC` (caller holds the lock)."""
        cutoff = time.monotonic() - UPLOAD_TTL_SEC
        for ref in [r for r, u in self.__uploads.items() if u.touched < cutoff]:
            del self.__uploads[ref]


__all__ = [
    "MAX_CHUNK_BYTES",
    "MAX_STAGED_BYTES",
    "MAX_UPLOADS",
    "UPLOAD_TTL_SEC",
    "Upload",
    "UploadError",
    "UploadRegistry",
]
