"""KV quantization at rest: a content-store layer that stores tensors smaller.

With ``--kv-quantization int8`` (or ``fp8_e4m3``, ``fp8_e5m2``, ``nf4``),
a payload that is a canonical KV frame (identity plus a tensor; see
:mod:`membrane.canonical`) is quantized before it reaches the store, and
dequantized back into a canonical frame of the original dtype and shape
when read. Quantization is lossy by design: it trades precision for
2-4x less memory, disk, and network. Payloads that are not tensor frames
pass through unchanged. Requires ``numpy`` (the ``transfer`` extra).

Stored layout of a quantized payload::

    b"MQV1" | identity length (u32 LE) | identity JSON | QuantizedFrame bytes
"""

import json
import logging
import struct
from typing import Any

from membrane.canonical import canonicalize, parse_canonical
from membrane.identity import PayloadIdentity

logger = logging.getLogger(__name__)

MAGIC = b"MQV1"
FORMATS = frozenset({"int8", "fp8_e4m3", "fp8_e5m2", "nf4"})
#: Tensor dtypes worth quantizing (floating point).
QUANTIZABLE_DTYPES = frozenset({"float16", "float32", "float64"})


class QuantizingStore:
    """Wraps a content store; quantizes canonical KV frames on the way in.

    Attributes:
        inner: The wrapped store.
        format_name: Quantization format.
    """

    def __init__(self, inner: Any, format_name: str) -> None:
        """Wrap ``inner``.

        Args:
            inner: The content store holding the (quantized) bytes.
            format_name: One of :data:`FORMATS`.

        Raises:
            ValueError: On an unknown format.
        """
        if format_name not in FORMATS:
            raise ValueError(f"unknown quantization format {format_name!r}; use one of {sorted(FORMATS)}")
        self.inner = inner
        self.format_name = format_name

    def encode(self, data: bytes) -> bytes:
        """Quantize ``data`` when it is a floating-point canonical frame.

        Args:
            data: Payload bytes.

        Returns:
            bytes: The quantized envelope, or ``data`` unchanged.
        """
        import numpy as np

        from membrane.quantization import quantize

        try:
            identity, raw = parse_canonical(data)
        except Exception:
            return data  # not a canonical frame
        if identity.dtype not in QUANTIZABLE_DTYPES:
            return data
        dtype = np.dtype(identity.dtype)
        count = 1
        for dim in identity.shape:
            count *= dim
        if count * dtype.itemsize != len(raw):
            return data  # payload does not match its declared shape
        tensor = np.frombuffer(raw, dtype=dtype).reshape(identity.shape)
        frame = quantize(tensor, self.format_name).to_bytes()
        identity_json = json.dumps(identity.to_dict(), sort_keys=True).encode()
        return MAGIC + struct.pack("<I", len(identity_json)) + identity_json + frame

    @staticmethod
    def decode(stored: bytes) -> bytes:
        """Turn a stored envelope back into a canonical frame.

        Args:
            stored: Bytes from the inner store.

        Returns:
            bytes: The canonical frame (dequantized), or ``stored`` when it
            is not a quantized envelope.
        """
        if not stored.startswith(MAGIC):
            return stored
        from membrane.quantization import QuantizedFrame, dequantize

        length = struct.unpack_from("<I", stored, len(MAGIC))[0]
        start = len(MAGIC) + 4
        identity = PayloadIdentity.from_dict(json.loads(stored[start : start + length]))
        tensor = dequantize(QuantizedFrame.from_bytes(stored[start + length :]))
        return canonicalize(identity, tensor.tobytes())

    def put(self, key: str, data: bytes) -> None:
        """Store ``data``, quantized when it is a tensor frame.

        Args:
            key: Content-store key.
            data: Payload bytes.
        """
        self.inner.put(key, self.encode(data))

    def get(self, key: str) -> bytes | None:
        """Read and dequantize.

        Args:
            key: Content-store key.

        Returns:
            bytes | None: The payload, or ``None`` when absent.
        """
        stored = self.inner.get(key)
        return None if stored is None else self.decode(stored)

    def has(self, key: str) -> bool:
        """Whether ``key`` is stored.

        Args:
            key: Content-store key.

        Returns:
            bool: True when present.
        """
        return bool(self.inner.has(key))

    def delete(self, key: str) -> bool:
        """Remove ``key``.

        Args:
            key: Content-store key.

        Returns:
            bool: True when something was removed.
        """
        return bool(self.inner.delete(key))

    def size(self) -> int:
        """Bytes held by the inner store (after quantization).

        Returns:
            int: Stored bytes.
        """
        return int(self.inner.size())

    def __getattr__(self, name: str) -> Any:
        """Expose the inner store's extras (``key_provider``, ``reencrypt_all``, ...).

        Args:
            name: Attribute name.

        Returns:
            Any: The inner store's attribute.
        """
        return getattr(self.inner, name)


__all__ = ["FORMATS", "MAGIC", "QUANTIZABLE_DTYPES", "QuantizingStore"]
