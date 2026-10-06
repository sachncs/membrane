"""Compression envelopes: lz4 (stand-in package), bombs, and malformed input."""

import struct
import sys
import types
import zlib

import pytest

from membrane.codec import CompressionTransport, method_available


@pytest.fixture
def fake_lz4(monkeypatch):
    """lz4.block's API and block format: a little-endian size prefix, then the data."""
    block = types.ModuleType("lz4.block")
    block.compress = lambda data: struct.pack("<I", len(data)) + zlib.compress(data)
    block.decompress = lambda data: zlib.decompress(data[4:])
    lz4 = types.ModuleType("lz4")
    lz4.block = block
    monkeypatch.setitem(sys.modules, "lz4", lz4)
    monkeypatch.setitem(sys.modules, "lz4.block", block)


def test_lz4_round_trip_and_declared_size_limit(fake_lz4) -> None:
    codec = CompressionTransport("lz4")
    wire = codec.compress(b"kv" * 1000)
    assert codec.decompress(wire) == b"kv" * 1000
    assert codec.decompress(wire, max_size=2000) == b"kv" * 1000
    with pytest.raises(ValueError, match="exceeds 100 bytes"):
        codec.decompress(wire, max_size=100)
    assert method_available("lz4")


def test_lz4_without_the_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lz4", None)
    monkeypatch.setitem(sys.modules, "lz4.block", None)
    assert not method_available("lz4")
    with pytest.raises(RuntimeError, match="requires the lz4 package"):
        CompressionTransport("lz4").compress(b"x")
    envelope = struct.pack("<BI", 4, 5) + b"12345"
    with pytest.raises(RuntimeError, match="requires the lz4 package"):
        CompressionTransport("raw").decompress(envelope)


@pytest.mark.parametrize("method", ["deflate", "zstd"])
def test_decompression_bombs_are_refused(method: str) -> None:
    codec = CompressionTransport(method)
    wire = codec.compress(b"\0" * 1_000_000)
    with pytest.raises(ValueError, match="exceeds 1000 bytes"):
        codec.decompress(wire, max_size=1000)
    assert codec.decompress(wire) == b"\0" * 1_000_000


def test_malformed_envelopes() -> None:
    codec = CompressionTransport("raw")
    with pytest.raises(ValueError, match="too short"):
        codec.decompress(b"\x01\x00")
    with pytest.raises(ValueError, match="too short"):
        codec.decompress(struct.pack("<BI", 1, 10) + b"abc")
    with pytest.raises(ValueError, match="unknown compression method id: 9"):
        codec.decompress(struct.pack("<BI", 9, 1) + b"x")
    with pytest.raises(ValueError, match="exceeds 2 bytes"):
        codec.decompress(codec.compress(b"abc"), max_size=2)
    assert not method_available("brotli")
