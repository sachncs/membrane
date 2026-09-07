"""GPU-aware memory management for KV transfers (Phase 4).

This module is the Phase 4 contract that ties the v2.0+ transfer
path together:

* :class:`MemoryPool` -- the abstract byte source / sink
  interface. The v1 of this arc ships two concrete pools:
  :class:`CudaMemoryPool` for GPU-resident tensors (with
  pinned host memory for staging) and :class:`RdmaMemoryPool`
  for cross-node GPU-to-GPU transfers where the hardware
  exposes RDMA (NCCL / GPUDirect Storage / libfabric).
* :class:`CompressionTransport` -- wraps the byte transport
  with optional zstd / lz4 compression. Operators that
  negotiate the slow path can pick compression; GPUDirect
  paths skip it.
* :class:`KVTransferEngine` -- a thin orchestrator that wires
  a :class:`KVAdapter` (Phase 2) + a quantizer (Phase 3) + a
  memory pool (this phase) into a single ``transfer_kv``
  call. Phase 5+ adapters can compose these into their engine
  integrations.
"""

from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory pool protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryPool(Protocol):
    """Abstract GPU / host memory pool."""

    def alloc(self, shape: tuple[int, ...], dtype: str) -> TensorHandle:
        """Allocate a tensor with the given shape and dtype."""
        ...

    def copy_to(self, src: TensorHandle, dst: TensorHandle) -> None:
        """Copy ``src`` into ``dst``."""
        ...

    def pin_host(self, src: TensorHandle) -> TensorHandle:
        """Pin ``src`` to host (page-locked) memory."""
        ...

    def close(self) -> None:
        """Release any resources held by the pool."""
        ...


class TensorHandle:
    """Opaque handle to a tensor in a :class:`MemoryPool`."""

    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return self.data

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# CUDA memory pool
# ---------------------------------------------------------------------------


class CudaMemoryPool:
    """GPU-resident memory pool with pinned host staging.

    Falls back to plain ``np.ndarray`` when torch CUDA is
    unavailable; the v1 of the tests runs in CPU-only mode.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._lock = threading.RLock()
        self._closed = False

    def alloc(self, shape: tuple[int, ...], dtype: str) -> TensorHandle:
        if self._closed:
            raise RuntimeError("CudaMemoryPool is closed")
        size = 1
        for d in shape:
            size *= d
        try:
            import torch

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Torch not compiled with CUDA enabled")
            tensor = torch.zeros(size, dtype=_torch_dtype(dtype), device=self.device)
            return TensorHandle(_torch_to_bytes(tensor), shape, dtype)
        except (ImportError, RuntimeError, AssertionError):
            bytes_payload = np.zeros(size, dtype=_numpy_dtype(dtype)).tobytes()
            return TensorHandle(bytes_payload, shape, dtype)

    def copy_to(self, src: TensorHandle, dst: TensorHandle) -> None:
        if len(dst.data) != len(src.data):
            raise ValueError("shape mismatch in copy_to")
        dst.data = src.data

    def pin_host(self, src: TensorHandle) -> TensorHandle:
        return TensorHandle(data=src.data, shape=src.shape, dtype=src.dtype)

    def close(self) -> None:
        with self._lock:
            self._closed = True


# ---------------------------------------------------------------------------
# RDMA memory pool
# ---------------------------------------------------------------------------


class RdmaMemoryPool:
    """GPUDirect / NCCL / libfabric-backed memory pool.

    The v1 implementation is a thin wrapper over
    :class:`CudaMemoryPool`. Operators on a GPUDirect-capable
    cluster install the wrapper and override the byte-level
    transport with their preferred library.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._delegate = CudaMemoryPool(device=device)
        self._lock = threading.RLock()
        self._closed = False

    def alloc(self, shape: tuple[int, ...], dtype: str) -> TensorHandle:
        return self._delegate.alloc(shape, dtype)

    def copy_to(self, src: TensorHandle, dst: TensorHandle) -> None:
        self._delegate.copy_to(src, dst)

    def pin_host(self, src: TensorHandle) -> TensorHandle:
        return self._delegate.pin_host(src)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._delegate.close()

    def rdma_send(self, src: TensorHandle, peer: str) -> int:
        """Stub for a future NCCL-based cross-node send."""
        logger.debug("RdmaMemoryPool.rdma_send stub: peer=%s size=%d", peer, src.size_bytes)
        return src.size_bytes


# ---------------------------------------------------------------------------
# Compression transport
# ---------------------------------------------------------------------------


class CompressionTransport:
    """Wraps the byte transport with optional zstd or lz4 compression.

    Wire format: 1-byte method id + 4-byte big-endian u32 length
    prefix + body. Method ids are 1=raw, 2=deflate, 3=zstd,
    4=lz4. The compressed body follows the length. The
    uncompressed size is not in the wire (operators can recover
    it from the underlying :class:`TensorHandle`); the u32
    length covers the compressed body only.
    """

    METHOD_RAW: str = "raw"
    METHOD_DEFLATE: str = "deflate"
    METHOD_ZSTD: str = "zstd"
    METHOD_LZ4: str = "lz4"

    _METHOD_IDS: ClassVar[dict[str, int]] = {
        METHOD_RAW: 1,
        METHOD_DEFLATE: 2,
        METHOD_ZSTD: 3,
        METHOD_LZ4: 4,
    }

    def __init__(self, method: str = METHOD_DEFLATE, level: int = 3) -> None:
        """Initialize the transport.

        Args:
            method: One of the ``METHOD_*`` constants.
            level: Compression level (1-9 for deflate, 1-22 for
                zstd). Ignored for raw and lz4.
        """
        if method not in self._METHOD_IDS:
            raise ValueError(f"unknown compression method: {method!r}")
        self.method = method
        self.level = level

    def compress(self, payload: bytes) -> bytes:
        """Compress ``payload`` with the configured method.

        Wire format: 1-byte method id + 4-byte u32 length + body.

        Args:
            payload: Raw bytes.

        Returns:
            bytes: The wire-format envelope.
        """
        if self.method == self.METHOD_RAW:
            body = payload
        elif self.method == self.METHOD_DEFLATE:
            import zlib

            body = zlib.compress(payload, self.level)
        elif self.method == self.METHOD_ZSTD:
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError(
                    "zstd compression requires the zstandard package"
                ) from exc
            compressor = zstandard.ZstdCompressor(level=self.level)
            body = compressor.compress(payload)
        else:  # lz4
            try:
                import lz4.block
            except ImportError as exc:
                raise RuntimeError(
                    "lz4 compression requires the lz4 package"
                ) from exc
            body = lz4.block.compress(payload)
        return struct.pack("<BI", self._METHOD_IDS[self.method], len(body)) + body

    def decompress(self, payload: bytes) -> bytes:
        """Inverse of :func:`compress`.

        Args:
            payload: Wire bytes from :func:`compress`.

        Returns:
            bytes: Decompressed bytes.
        """
        if len(payload) < 5:
            raise ValueError("compressed payload too short")
        method_id = struct.unpack_from("<B", payload, 0)[0]
        body_len = struct.unpack_from("<I", payload, 1)[0]
        if len(payload) < 5 + body_len:
            raise ValueError("compressed payload too short")
        body = payload[5 : 5 + body_len]
        if method_id == 1:
            return body
        if method_id == 2:
            import zlib

            return zlib.decompress(body)
        if method_id == 3:
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError(
                    "zstd decompression requires the zstandard package"
                ) from exc
            return zstandard.ZstdDecompressor().decompress(body)
        if method_id == 4:
            try:
                import lz4.block
            except ImportError as exc:
                raise RuntimeError(
                    "lz4 decompression requires the lz4 package"
                ) from exc
            return lz4.block.decompress(body)
        raise ValueError(f"unknown compression method id: {method_id}")


# ---------------------------------------------------------------------------
# KV transfer engine
# ---------------------------------------------------------------------------


class KVTransferEngine:
    """Compose KVAdapter + quantizer + memory pool + transport."""

    def __init__(
        self,
        memory_pool: MemoryPool,
        transport: CompressionTransport | None = None,
        quantizer: Any | None = None,
    ) -> None:
        self.memory_pool = memory_pool
        self.transport = transport or CompressionTransport(
            method=CompressionTransport.METHOD_RAW
        )
        self.quantizer = quantizer

    def transfer_kv(
        self,
        k_handle: TensorHandle,
        v_handle: TensorHandle,
    ) -> TransferEnvelope:
        """Bundle ``k_handle`` and ``v_handle`` into a :class:`TransferEnvelope`.

        The transfer is wrapped in an OTel ``transfer.kv`` span
        when the tracer is configured; absent the tracer the
        function returns the envelope unchanged.
        """
        from membrane.otel_tracer import membrane_span

        raw_k = k_handle.tobytes()
        raw_v = v_handle.tobytes()
        if self.quantizer is not None:
            k_frame = self.quantizer.quantize(
                _bytes_to_array(raw_k, k_handle.shape, k_handle.dtype)
            )
            v_frame = self.quantizer.quantize(
                _bytes_to_array(raw_v, v_handle.shape, v_handle.dtype)
            )
            raw_k = k_frame.to_bytes() if hasattr(k_frame, "to_bytes") else k_frame
            raw_v = v_frame.to_bytes() if hasattr(v_frame, "to_bytes") else v_frame
        payload = (
            b"MKVR"
            + struct.pack("<I", len(raw_k))
            + struct.pack("<I", len(raw_v))
            + raw_k
            + raw_v
        )
        with membrane_span(
            "transfer.kv",
            kv_bytes=str(len(payload)),
            compression=self.transport.method,
            shape="x".join(str(d) for d in k_handle.shape),
            dtype=k_handle.dtype,
        ):
            compressed = self.transport.compress(payload)
        return TransferEnvelope(
            compressed=compressed,
            compression=self.transport.method,
            shape=k_handle.shape,
            dtype=k_handle.dtype,
        )

    def receive_kv(
        self,
        envelope: TransferEnvelope,
    ) -> tuple[TensorHandle, TensorHandle]:
        """Inverse of :func:`transfer_kv`."""
        payload = self.transport.decompress(envelope.compressed)
        if not payload.startswith(b"MKVR"):
            raise ValueError(
                f"bad magic in transfer envelope: {payload[:4]!r}"
            )
        offset = 4
        k_len = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        v_len = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        k_handle = TensorHandle(
            payload[offset : offset + k_len], envelope.shape, envelope.dtype
        )
        offset += k_len
        v_handle = TensorHandle(
            payload[offset : offset + v_len], envelope.shape, envelope.dtype
        )
        return k_handle, v_handle


@dataclass(frozen=True)
class TransferEnvelope:
    """Compressed wire bundle produced by :class:`KVTransferEngine`."""

    compressed: bytes
    compression: str
    shape: tuple[int, ...]
    dtype: str


__all__ = [
    "CompressionTransport",
    "CudaMemoryPool",
    "KVTransferEngine",
    "MemoryPool",
    "RdmaMemoryPool",
    "TensorHandle",
    "TransferEnvelope",
]


def _numpy_dtype(name: str) -> Any:
    return np.dtype(name)


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _torch_to_bytes(tensor: Any) -> bytes:
    return tensor.detach().cpu().numpy().tobytes()


def _bytes_to_array(payload: bytes, shape: tuple[int, ...], dtype: str) -> Any:
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)
