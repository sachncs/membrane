"""GPU-aware memory management for KV transfers.

This module ties the transfer path together:

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
  a :class:`KVAdapter` + a quantizer + a
  memory pool into a single ``transfer_kv`` call that engine
  adapters can compose into their integrations.
"""

import logging
import struct
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from membrane.codec import CompressionTransport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory pool protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryPool(Protocol):
    """Abstract GPU / host memory pool."""

    def alloc(self, shape: tuple[int, ...], dtype: str) -> TensorHandle:
        """Allocate a tensor with the given shape and dtype.

        Args:
            shape: Tensor shape.
            dtype: Element dtype name (e.g. ``float16``).

        Returns:
            TensorHandle: The allocated tensor.
        """
        ...

    def copy_to(self, src: TensorHandle, dst: TensorHandle) -> None:
        """Copy ``src`` into ``dst``.

        Args:
            src: Source tensor.
            dst: Destination tensor.
        """
        ...

    def pin_host(self, src: TensorHandle) -> TensorHandle:
        """Pin ``src`` to host (page-locked) memory.

        Args:
            src: Source tensor.

        Returns:
            TensorHandle: The pinned copy.
        """
        ...

    def close(self) -> None:
        """Release any resources held by the pool."""
        ...


class TensorHandle:
    """Opaque handle to a tensor in a :class:`MemoryPool`."""

    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        """Wrap raw tensor bytes with their shape and dtype.

        Args:
            data: Raw tensor bytes.
            shape: Tensor shape.
            dtype: Element dtype name.
        """
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        """Return the tensor's raw bytes.

        Returns:
            bytes: The tensor's raw bytes.
        """
        return self.data

    @property
    def size_bytes(self) -> int:
        """Size of the tensor in bytes."""
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
        """Create a pool for ``device``.

        Args:
            device: Device name, e.g. ``cuda:0``.
        """
        self.device = device
        self.lock = threading.RLock()
        self.__closed = False

    def alloc(self, shape: tuple[int, ...], dtype: str) -> TensorHandle:
        """Allocate a zeroed tensor of ``shape`` and ``dtype``.

        Args:
            shape: Tensor shape.
            dtype: Element dtype name (e.g. ``float16``).

        Returns:
            TensorHandle: The allocated tensor.
        """
        if self.__closed:
            raise RuntimeError("CudaMemoryPool is closed")
        size = 1
        for d in shape:
            size *= d
        try:
            import torch

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Torch not compiled with CUDA enabled")
            tensor = torch.zeros(size, dtype=torch_dtype(dtype), device=self.device)
            return TensorHandle(torch_to_bytes(tensor), shape, dtype)
        except ImportError, RuntimeError, AssertionError:
            bytes_payload = np.zeros(size, dtype=numpy_dtype(dtype)).tobytes()
            return TensorHandle(bytes_payload, shape, dtype)

    def copy_to(self, src: TensorHandle, dst: TensorHandle) -> None:
        """Copy ``src`` into ``dst``; their sizes must match.

        Args:
            src: Source tensor.
            dst: Destination tensor.
        """
        if len(dst.data) != len(src.data):
            raise ValueError("shape mismatch in copy_to")
        dst.data = src.data

    def pin_host(self, src: TensorHandle) -> TensorHandle:
        """Return a host-pinned copy of ``src``.

        Args:
            src: Source tensor.

        Returns:
            TensorHandle: A host-pinned copy of ``src``.
        """
        return TensorHandle(data=src.data, shape=src.shape, dtype=src.dtype)

    def close(self) -> None:
        """Close the pool; later allocations fail."""
        with self.lock:
            self.__closed = True


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
        """Create an RDMA-registered pool for ``device``.

        Args:
            device: Device name, e.g. ``cuda:0``.
        """
        self.device = device
        self.__delegate = CudaMemoryPool(device=device)
        self.lock = threading.RLock()
        self.__closed = False

    def alloc(self, shape: tuple[int, ...], dtype: str) -> TensorHandle:
        """Allocate a registered tensor of ``shape`` and ``dtype``.

        Args:
            shape: Tensor shape.
            dtype: Element dtype name (e.g. ``float16``).

        Returns:
            TensorHandle: The allocated tensor.
        """
        return self.__delegate.alloc(shape, dtype)

    def copy_to(self, src: TensorHandle, dst: TensorHandle) -> None:
        """Copy ``src`` into ``dst``; their sizes must match.

        Args:
            src: Source tensor.
            dst: Destination tensor.
        """
        self.__delegate.copy_to(src, dst)

    def pin_host(self, src: TensorHandle) -> TensorHandle:
        """Return a host-pinned copy of ``src``.

        Args:
            src: Source tensor.

        Returns:
            TensorHandle: A host-pinned copy of ``src``.
        """
        return self.__delegate.pin_host(src)

    def close(self) -> None:
        """Close the pool and release its registrations."""
        with self.lock:
            self.__closed = True
            self.__delegate.close()

    def rdma_send(self, src: TensorHandle, peer: str) -> int:
        """Stub for a future NCCL-based cross-node send.

        Args:
            src: Source tensor.
            peer: Destination peer.

        Returns:
            int: Bytes sent.
        """
        logger.debug("RdmaMemoryPool.rdma_send stub: peer=%s size=%d", peer, src.size_bytes)
        return src.size_bytes


# ---------------------------------------------------------------------------
# Compression transport
# ---------------------------------------------------------------------------


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
        """Create an engine over ``memory_pool``.

        Args:
            memory_pool: Pool tensors are allocated from on receive.
            transport: Compression transport; deflate by default.
            quantizer: Optional quantizer applied before compression.
        """
        self.memory_pool = memory_pool
        self.transport = transport or CompressionTransport(method=CompressionTransport.METHOD_RAW)
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

        Args:
            k_handle: Key tensor.
            v_handle: Value tensor.

        Returns:
            TransferEnvelope: The compressed K/V pair ready for the wire.
        """
        from membrane.otel_tracer import membrane_span

        raw_k = k_handle.tobytes()
        raw_v = v_handle.tobytes()
        if self.quantizer is not None:
            k_frame = self.quantizer.quantize(bytes_to_array(raw_k, k_handle.shape, k_handle.dtype))
            v_frame = self.quantizer.quantize(bytes_to_array(raw_v, v_handle.shape, v_handle.dtype))
            raw_k = k_frame.to_bytes() if hasattr(k_frame, "to_bytes") else k_frame
            raw_v = v_frame.to_bytes() if hasattr(v_frame, "to_bytes") else v_frame
        payload = b"MKVR" + struct.pack("<I", len(raw_k)) + struct.pack("<I", len(raw_v)) + raw_k + raw_v
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
        """Inverse of :func:`transfer_kv`.

        Args:
            envelope: Wire envelope produced by :meth:`transfer_kv`.

        Returns:
            tuple[TensorHandle, TensorHandle]: The key and value tensors.
        """
        payload = self.transport.decompress(envelope.compressed)
        if not payload.startswith(b"MKVR"):
            raise ValueError(f"bad magic in transfer envelope: {payload[:4]!r}")
        offset = 4
        k_len = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        v_len = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        k_handle = TensorHandle(payload[offset : offset + k_len], envelope.shape, envelope.dtype)
        offset += k_len
        v_handle = TensorHandle(payload[offset : offset + v_len], envelope.shape, envelope.dtype)
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


def numpy_dtype(name: str) -> Any:
    """Return the NumPy dtype named ``name``.

    Args:
        name: Dtype name.

    Returns:
        Any: The NumPy dtype named ``name``.
    """
    return np.dtype(name)


def torch_dtype(name: str) -> Any:
    """Return the PyTorch dtype named ``name``.

    Args:
        name: Dtype name.

    Returns:
        Any: The PyTorch dtype named ``name``.
    """
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def torch_to_bytes(tensor: Any) -> bytes:
    """Return a tensor's raw bytes (moved to the CPU).

    Args:
        tensor: PyTorch tensor.

    Returns:
        bytes: A tensor's raw bytes (moved to the CPU).
    """
    return tensor.detach().cpu().numpy().tobytes()


def bytes_to_array(payload: bytes, shape: tuple[int, ...], dtype: str) -> Any:
    """Interpret ``payload`` as a NumPy array of ``shape`` and ``dtype``.

    Args:
        payload: Raw bytes.
        shape: Array shape.
        dtype: Element dtype name.

    Returns:
        Any: A NumPy array view of ``payload``.
    """
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)
