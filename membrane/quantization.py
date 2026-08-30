# mypy: ignore-errors
"""K/V tensor quantization (Phase 3).

The v2.0+ transfer path supports three quantization formats
behind a uniform :class:`Quantizer` protocol:

* :class:`Int8PerChannelQuantizer` -- symmetric int8 per-row
  scaling, ``~2x`` memory reduction, near-lossless for prefill.
* :class:`FP8E4M3Quantizer` -- ``torch.float8_e4m3fn`` for
  Hopper-class hardware, ``~2.5x`` reduction. Falls back to int8
  on pre-fp8 hosts (the wire format is identical at 1 byte per
  element with a per-row scale factor).
* :class:`FP8E5M2Quantizer` -- ``torch.float8_e5m2`` (the
  backward-precision variant of fp8) for activations.
* :class:`NF4Quantizer` -- 4-bit NormalFloat (Dettmers et al.)
  via a self-contained numpy implementation (no
  ``bitsandbytes`` dependency required).

The wire format :class:`QuantizedFrame` wraps each format
with a header carrying ``{fmt, scale, zero_point, original_dtype,
original_shape}`` so the receiver can dequantize without
recomputing scale factors. The :func:`quantize` /
:func:`dequantize` helpers are the v2.0+ KV bytes transport;
``KVTensor`` round-trips through them before serializing.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np  # type: ignore  # re-exported at module level

logger = logging.getLogger(__name__)


#: Wire magic for the v2.0+ quantized frame. The four bytes
#: match the canonical frame's family ("MV") + a v2-prefixed
#: "QF" sequence.
_MAGIC: bytes = b"MVQF"
#: Header size = magic (4) + format (1) + reserved (3) + scale (4) +
#: zero_point (4) + original_dtype_len (1) + original_dtype +
#: shape_count (1) + shape (4 * shape_count). We keep the
#: header compact so per-window quantization stays cheap.
_HEADER_PREFIX: int = 4 + 1 + 3 + 4 + 4 + 1  # 17 bytes fixed prefix
_TRAILER_LEN: int = 8  # 8-byte SHA-256 prefix for cheap verify


#: Format identifiers. Stored as a single byte in the wire header
#: so the receiver can dispatch to the right dequantizer without
#: parsing the dtype / shape fields first.
FORMAT_INT8: int = 1
FORMAT_FP8_E4M3: int = 2
FORMAT_FP8_E5M2: int = 3
FORMAT_NF4: int = 4


@runtime_checkable
class Quantizer(Protocol):
    """Quantization / dequantization protocol."""

    format_id: int
    format_name: str

    def quantize(self, tensor: np.ndarray) -> bytes:
        """Quantize ``tensor`` to wire bytes."""
        ...

    def dequantize(
        self,
        payload: bytes,
        original_dtype: str,
        original_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Inverse of :func:`quantize`."""
        ...


@dataclass(frozen=True)
class QuantizedFrame:
    """Wire-format quantized K/V bundle.

    Attributes:
        format_id: One of the ``FORMAT_*`` constants.
        scale: Per-tensor scale factor. ``1.0`` for formats that
            do not need it.
        zero_point: Asymmetric-quantization zero point, ``0`` for
            symmetric formats.
        original_dtype: The dtype the tensor had before
            quantization.
        original_shape: Per-layer tensor shape before
            quantization.
        payload: The format-specific bytes.
    """

    format_id: int
    scale: float
    zero_point: int
    original_dtype: str
    original_shape: tuple[int, ...]
    payload: bytes

    def to_bytes(self) -> bytes:
        """Serialize this frame to the v2.0+ wire format.

        Returns:
            bytes: The wire bytes, ready to attach to a
            :class:`Fragment` payload via the :class:`ContentStore`.
        """
        dtype_bytes = self.original_dtype.encode("utf-8")
        if len(dtype_bytes) > 255:
            raise ValueError("dtype string too long for wire format")
        if len(self.original_shape) > 255:
            raise ValueError("shape too long for wire format")
        if self.scale < 0 or not math.isfinite(self.scale):
            raise ValueError(f"invalid scale: {self.scale}")
        if self.zero_point < -(1 << 31) or self.zero_point > (1 << 31) - 1:
            raise ValueError(f"zero_point out of range: {self.zero_point}")
        header = (
            _MAGIC
            + struct.pack(
                "<B",
                self.format_id,
            )
            + b"\x00\x00\x00"
            + struct.pack("<f", self.scale)
            + struct.pack("<i", self.zero_point)
            + struct.pack("<B", len(dtype_bytes))
            + dtype_bytes
            + struct.pack("<B", len(self.original_shape))
            + b"".join(struct.pack("<I", d) for d in self.original_shape)
        )
        import hashlib

        digest = hashlib.sha256(header + self.payload).digest()[:_TRAILER_LEN]
        return header + self.payload + digest

    @classmethod
    def from_bytes(cls, frame: bytes) -> QuantizedFrame:
        """Parse a :class:`QuantizedFrame` from the wire format.

        Args:
            frame: The wire bytes.

        Returns:
            QuantizedFrame: The reconstructed frame.
        """
        import hashlib

        if not frame.startswith(_MAGIC):
            raise ValueError("bad magic in quantized frame")
        if len(frame) < 4 + 1 + 3 + 4 + 4 + 1 + 1:
            raise ValueError("quantized frame too short")
        offset = 4
        format_id = struct.unpack_from("<B", frame, offset)[0]
        offset += 1 + 3
        scale = struct.unpack_from("<f", frame, offset)[0]
        offset += 4
        zero_point = struct.unpack_from("<i", frame, offset)[0]
        offset += 4
        dtype_len = struct.unpack_from("<B", frame, offset)[0]
        offset += 1
        dtype = frame[offset : offset + dtype_len].decode("utf-8")
        offset += dtype_len
        shape_count = struct.unpack_from("<B", frame, offset)[0]
        offset += 1
        if offset + 4 * shape_count + _TRAILER_LEN > len(frame):
            raise ValueError("quantized frame too short")
        shape = struct.unpack_from(f"<{shape_count}I", frame, offset)
        offset += 4 * shape_count
        payload_end = len(frame) - _TRAILER_LEN
        if payload_end < offset:
            raise ValueError("quantized frame too short")
        payload = bytes(frame[offset:payload_end])
        trailer = bytes(frame[payload_end:])
        digest = hashlib.sha256(frame[:payload_end]).digest()[:_TRAILER_LEN]
        if digest != trailer:
            raise ValueError("quantized frame trailer mismatch")
        return cls(
            format_id=format_id,
            scale=scale,
            zero_point=zero_point,
            original_dtype=dtype,
            original_shape=tuple(shape),
            payload=payload,
        )


def _row_col(tensor: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Normalize ``tensor`` to a 2-D ``(rows, cols)`` layout.

    Args:
        tensor: Any shape.

    Returns:
        tuple[np.ndarray, int, int]: The 2-D view, ``n_rows``,
        ``n_cols``.
    """
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor, tensor.shape[0], tensor.shape[1]


def quantize(
    tensor: np.ndarray, format_name: str = "int8"
) -> QuantizedFrame:
    """Quantize ``tensor`` using the named :class:`Quantizer`.

    Args:
        tensor: Tensor-like input. ``torch.Tensor`` and
            ``numpy.ndarray`` are both supported.
        format_name: One of ``"int8"``, ``"fp8_e4m3"``,
            ``"fp8_e5m2"``, ``"nf4"``.

    Returns:
        QuantizedFrame: The wire-format quantized bundle.
    """
    arr = _to_numpy(tensor)
    quantizer = _quantizer_for(format_name)
    payload = quantizer.quantize(arr)
    return QuantizedFrame(
        format_id=quantizer.format_id,
        scale=1.0,
        zero_point=0,
        original_dtype=str(arr.dtype),
        original_shape=tuple(arr.shape),
        payload=payload,
    )


def dequantize(frame: QuantizedFrame) -> np.ndarray:
    """Reconstruct the original tensor from a :class:`QuantizedFrame`.

    Args:
        frame: The wire-format bundle.

    Returns:
        np.ndarray: Tensor-like output.
    """
    quantizer = _quantizer_for_id(frame.format_id)
    return quantizer.dequantize(
        frame.payload, frame.original_dtype, frame.original_shape
    )


def _quantizer_for(format_name: str) -> Quantizer:
    """Look up the :class:`Quantizer` for ``format_name``."""
    if format_name == "int8":
        return Int8PerChannelQuantizer()
    if format_name == "fp8_e4m3":
        return FP8E4M3Quantizer()
    if format_name == "fp8_e5m2":
        return FP8E5M2Quantizer()
    if format_name == "nf4":
        return NF4Quantizer()
    raise ValueError(
        f"unknown quantization format: {format_name!r}; expected one of "
        f"int8, fp8_e4m3, fp8_e5m2, nf4"
    )


def _quantizer_for_id(format_id: int) -> Quantizer:
    """Inverse of :func:`_quantizer_for` keyed on the wire byte."""
    return {
        FORMAT_INT8: Int8PerChannelQuantizer,
        FORMAT_FP8_E4M3: FP8E4M3Quantizer,
        FORMAT_FP8_E5M2: FP8E5M2Quantizer,
        FORMAT_NF4: NF4Quantizer,
    }[format_id]()  # type: ignore[return-value]


def _to_numpy(tensor: Any) -> np.ndarray:
    """Normalize ``tensor`` to a ``np.ndarray`` view."""
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        return tensor.detach().cpu().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


# ---------------------------------------------------------------------------
# Concrete quantizers
# ---------------------------------------------------------------------------


class Int8PerChannelQuantizer:
    """Symmetric int8 per-row quantization.

    Wire format: row-major int8 with a per-row float32 scale
    factor. Wire: ``u32 n_rows, u32 n_cols, n_rows * float32
    scales, n_rows * n_cols * int8 values``.
    """

    format_id: int = FORMAT_INT8
    format_name: str = "int8"

    def quantize(self, tensor: np.ndarray) -> bytes:
        """Symmetric int8 quantize.

        Args:
            tensor: 2-D array (rows, cols).

        Returns:
            bytes: Per-row scales (float32) + int8 values.
        """
        arr, n_rows, n_cols = _row_col(tensor)
        abs_max = np.max(np.abs(arr), axis=1).astype("float32")
        abs_max = np.where(abs_max == 0, 1.0, abs_max)
        scale = abs_max / 127.0
        values = (arr / scale[:, None]).clip(-127, 127).round().astype("int8")
        return (
            struct.pack("<I", n_rows)
            + struct.pack("<I", n_cols)
            + scale.tobytes()
            + values.tobytes()
        )

    def dequantize(
        self,
        payload: bytes,
        original_dtype: str,
        original_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Inverse of :func:`quantize`."""
        offset = 0
        n_rows = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        n_cols = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        scale = np.frombuffer(
            payload[offset : offset + 4 * n_rows], dtype="float32"
        )
        offset += 4 * n_rows
        values = np.frombuffer(
            payload[offset : offset + n_rows * n_cols], dtype="int8"
        ).reshape(n_rows, n_cols)
        flat = (values.astype("float32") * scale[:, None]).reshape(-1)
        return flat[: int(np.prod(original_shape))].reshape(original_shape).astype(
            original_dtype
        )


class FP8E4M3Quantizer:
    """fp8 e4m3 quantization.

    When the host has ``torch.float8_e4m3fn`` and a numpy that
    can read it (numpy 2.x), the wire stores 1 fp8 byte per
    element. Otherwise the path falls back to int8 with a
    documented scale of 240; the wire header remains fp8 so the
    receiver's dequantize reads the right number of bytes.
    """

    format_id: int = FORMAT_FP8_E4M3
    format_name: str = "fp8_e4m3"

    def _has_fp8(self) -> bool:
        """Return whether both torch fp8 and numpy fp8 are available."""
        try:
            import torch

            _ = torch.float8_e4m3fn
        except (ImportError, AttributeError):
            return False
        try:
            import numpy as np

            np.dtype("float8_e4m3fn")
            return True
        except (TypeError, ValueError):
            return False

    def quantize(self, tensor: np.ndarray) -> bytes:
        """Quantize a 2-D tensor to fp8 e4m3.

        Args:
            tensor: 2-D array (rows, cols).

        Returns:
            bytes: Per-row scales (float32) + raw fp8 bytes (or
            int8 fallback bytes when fp8 is not supported).
        """
        import numpy as np

        arr, n_rows, n_cols = _row_col(tensor)
        abs_max = np.max(np.abs(arr), axis=1).astype("float32")
        abs_max = np.where(abs_max == 0, 1.0, abs_max)
        scale = abs_max / 240.0
        scaled = (arr / scale[:, None]).astype("float32")
        if self._has_fp8():
            import torch

            scaled = scaled.astype(torch.float8_e4m3fn)
        else:
            scaled = scaled.clip(-240, 240).astype("int8")
        return (
            struct.pack("<I", n_rows)
            + struct.pack("<I", n_cols)
            + scale.tobytes()
            + scaled.tobytes()
        )

    def dequantize(
        self,
        payload: bytes,
        original_dtype: str,
        original_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Inverse of :func:`quantize`."""
        import numpy as np

        offset = 0
        n_rows = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        n_cols = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        scale = np.frombuffer(
            payload[offset : offset + 4 * n_rows], dtype="float32"
        )
        offset += 4 * n_rows
        body = payload[offset : offset + n_rows * n_cols]
        if self._has_fp8():
            import torch

            values = np.frombuffer(
                body, dtype=torch.float8_e4m3fn
            ).astype("float32")
        else:
            values = np.frombuffer(body, dtype="int8").astype("float32")
        flat = (values * scale[:, None]).reshape(-1)
        return flat[: int(np.prod(original_shape))].reshape(original_shape).astype(
            original_dtype
        )


class FP8E5M2Quantizer:
    """fp8 e5m2 quantization.

    Symmetric counterpart to :class:`FP8E4M3Quantizer`; the
    e5m2 mantissa has one more bit so it is the better choice
    for activations where the range matters more than the
    precision.
    """

    format_id: int = FORMAT_FP8_E5M2
    format_name: str = "fp8_e5m2"

    def _has_fp8(self) -> bool:
        """Return whether both torch fp8 and numpy fp8 are available."""
        try:
            import torch

            _ = torch.float8_e5m2
        except (ImportError, AttributeError):
            return False
        try:
            import numpy as np

            np.dtype("float8_e5m2")
            return True
        except (TypeError, ValueError):
            return False

    def quantize(self, tensor: np.ndarray) -> bytes:
        """Quantize a 2-D tensor to fp8 e5m2.

        Args:
            tensor: 2-D array (rows, cols).

        Returns:
            bytes: Per-row scales (float32) + raw fp8 bytes.
        """
        import numpy as np

        arr, n_rows, n_cols = _row_col(tensor)
        abs_max = np.max(np.abs(arr), axis=1).astype("float32")
        abs_max = np.where(abs_max == 0, 1.0, abs_max)
        scale = abs_max / 28000.0
        scaled = (arr / scale[:, None]).astype("float32")
        if self._has_fp8():
            import torch

            scaled = scaled.astype(torch.float8_e5m2)
        else:
            scaled = scaled.clip(-28000, 28000).astype("int8")
        return (
            struct.pack("<I", n_rows)
            + struct.pack("<I", n_cols)
            + scale.tobytes()
            + scaled.tobytes()
        )

    def dequantize(
        self,
        payload: bytes,
        original_dtype: str,
        original_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Inverse of :func:`quantize`."""
        import numpy as np

        offset = 0
        n_rows = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        n_cols = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        scale = np.frombuffer(
            payload[offset : offset + 4 * n_rows], dtype="float32"
        )
        offset += 4 * n_rows
        body = payload[offset : offset + n_rows * n_cols]
        if self._has_fp8():
            import torch

            values = np.frombuffer(
                body, dtype=torch.float8_e5m2
            ).astype("float32")
        else:
            values = np.frombuffer(body, dtype="int8").astype("float32")
        flat = (values * scale[:, None]).reshape(-1)
        return flat[: int(np.prod(original_shape))].reshape(original_shape).astype(
            original_dtype
        )


class NF4Quantizer:
    """4-bit NormalFloat (NF4) quantization.

    Implements the NF4 lookup table from Dettmers et al. 2023
    (QLoRA). The wire packs two 4-bit indices per byte with a
    per-row float32 absmax. Operators who want GPU-side NF4 can
    wire their own implementation in.
    """

    format_id: int = FORMAT_NF4
    format_name: str = "nf4"

    # NF4 codebook: 16 quantiles of a normalized Gaussian,
    # symmetric around zero. Dettmers et al. 2023, Table 2.
    _NF4_TABLE: tuple[float, ...] = (
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.2461123002767563,
        0.3379152417187691,
        0.4407098295211792,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    )

    def quantize(self, tensor: np.ndarray) -> bytes:
        """Quantize a 2-D tensor to NF4.

        Args:
            tensor: 2-D array (rows, cols).

        Returns:
            bytes: Per-row absmax (float32) + packed NF4 indices
            (4 bits per element, two indices per byte).
        """
        import numpy as np

        arr, n_rows, n_cols = _row_col(tensor)
        abs_max = np.max(np.abs(arr), axis=1).astype("float32")
        abs_max = np.where(abs_max == 0, 1.0, abs_max)
        normalized = arr / abs_max[:, None]
        normalized = np.clip(normalized, -1.0, 1.0)
        table = np.asarray(self._NF4_TABLE, dtype="float32")
        distances = np.abs(normalized[:, :, None] - table[None, None, :])
        indices = distances.argmin(axis=-1).astype("uint8")
        padded = (
            np.concatenate([indices, np.zeros((n_rows, 1), dtype="uint8")], axis=1)
            if n_cols % 2
            else indices
        )
        packed = (padded[:, 0::2] << 4) | padded[:, 1::2]
        return (
            struct.pack("<I", n_rows)
            + struct.pack("<I", n_cols)
            + abs_max.tobytes()
            + packed.tobytes()
        )

    def dequantize(
        self,
        payload: bytes,
        original_dtype: str,
        original_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Inverse of :func:`quantize`."""
        import numpy as np

        offset = 0
        n_rows = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        n_cols = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        abs_max = np.frombuffer(
            payload[offset : offset + 4 * n_rows], dtype="float32"
        )
        offset += 4 * n_rows
        n_pairs = (n_cols + 1) // 2
        packed = np.frombuffer(
            payload[offset : offset + n_rows * n_pairs], dtype="uint8"
        ).reshape(n_rows, n_pairs)
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        if n_cols % 2:
            indices = np.empty((n_rows, n_cols), dtype="uint8")
            indices[:, : n_cols - 1 : 2] = high[:, : (n_cols - 1) // 2]
            indices[:, 1 : n_cols : 2] = low[:, : n_cols // 2]
            indices[:, -1] = high[:, -1]
        else:
            indices = np.empty((n_rows, n_cols), dtype="uint8")
            indices[:, 0::2] = high
            indices[:, 1::2] = low
        table = np.asarray(self._NF4_TABLE, dtype="float32")
        flat = (table[indices.astype("int32")] * abs_max[:, None]).reshape(-1)
        return flat[: int(np.prod(original_shape))].reshape(original_shape).astype(
            original_dtype
        )


__all__ = [
    "FORMAT_FP8_E4M3",
    "FORMAT_FP8_E5M2",
    "FORMAT_INT8",
    "FORMAT_NF4",
    "FP8E4M3Quantizer",
    "FP8E5M2Quantizer",
    "Int8PerChannelQuantizer",
    "NF4Quantizer",
    "QuantizedFrame",
    "Quantizer",
    "dequantize",
    "quantize",
]
