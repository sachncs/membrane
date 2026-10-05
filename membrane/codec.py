"""Byte compression for KV payloads moving between nodes.

:class:`CompressionTransport` frames a body as ``method id (1 byte) +
length (u32 LE) + compressed body``. zstd comes from the standard
library (PEP 784); lz4 needs the ``lz4`` package (``transfer`` extra) and
has no free-threaded wheel, so prefer zstd there. Decompression can be
bounded (``max_size``) for bodies from untrusted senders.
"""

import struct
from compression import zstd
from typing import ClassVar


class CompressionTransport:
    """Wraps the byte transport with optional zstd or lz4 compression.

    Wire format: 1-byte method id + 4-byte little-endian u32 length
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

    __METHOD_IDS: ClassVar[dict[str, int]] = {
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
        if method not in self.__METHOD_IDS:
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
            # Standard-library Zstandard (PEP 784); no third-party package.
            body = zstd.compress(payload, level=self.level)
        else:  # lz4
            try:
                import lz4.block
            except ImportError as exc:
                raise RuntimeError("lz4 compression requires the lz4 package") from exc
            body = lz4.block.compress(payload)
        return struct.pack("<BI", self.__METHOD_IDS[self.method], len(body)) + body

    def decompress(self, payload: bytes, max_size: int | None = None) -> bytes:
        """Inverse of :func:`compress`, optionally bounded.

        Args:
            payload: Wire bytes from :func:`compress`.
            max_size: Refuse output larger than this many bytes (guards
                against decompression bombs from untrusted senders);
                unbounded when ``None``.

        Returns:
            bytes: Decompressed bytes.

        Raises:
            ValueError: On a malformed envelope, an unknown method, or
                output above ``max_size``.
        """
        if len(payload) < 5:
            raise ValueError("compressed payload too short")
        method_id = struct.unpack_from("<B", payload, 0)[0]
        body_len = struct.unpack_from("<I", payload, 1)[0]
        if len(payload) < 5 + body_len:
            raise ValueError("compressed payload too short")
        body = payload[5 : 5 + body_len]
        limit = -1 if max_size is None else max_size + 1
        match method_id:
            case 1:
                out = body
            case 2:
                import zlib

                inflater = zlib.decompressobj()
                out = inflater.decompress(body, 0 if limit < 0 else limit)
                if limit >= 0 and inflater.unconsumed_tail:
                    raise ValueError(f"decompressed payload exceeds {max_size} bytes")
            case 3:
                decompressor = zstd.ZstdDecompressor()
                out = decompressor.decompress(body, max_length=limit)
                if limit >= 0 and not decompressor.eof:
                    raise ValueError(f"decompressed payload exceeds {max_size} bytes")
            case 4:
                try:
                    import lz4.block
                except ImportError as exc:
                    raise RuntimeError("lz4 decompression requires the lz4 package") from exc
                # lz4 block frames start with the uncompressed size.
                declared = struct.unpack_from("<I", body, 0)[0] if len(body) >= 4 else 0
                if max_size is not None and declared > max_size:
                    raise ValueError(f"decompressed payload exceeds {max_size} bytes")
                out = lz4.block.decompress(body)
            case _:
                raise ValueError(f"unknown compression method id: {method_id}")
        if max_size is not None and len(out) > max_size:
            raise ValueError(f"decompressed payload exceeds {max_size} bytes")
        return out


#: Methods a node may select for peer transfers.
COMPRESSION_METHODS = frozenset({"zstd", "lz4", "deflate", "raw"})


def method_available(method: str) -> bool:
    """Whether ``method`` can be used in this environment.

    Args:
        method: A :data:`COMPRESSION_METHODS` name.

    Returns:
        bool: False for unknown methods, and for ``lz4`` without the
        ``lz4`` package.
    """
    if method not in COMPRESSION_METHODS:
        return False
    if method == "lz4":
        try:
            import lz4.block  # noqa: F401  -- availability probe
        except ImportError:
            return False
    return True


__all__ = ["COMPRESSION_METHODS", "CompressionTransport", "method_available"]
