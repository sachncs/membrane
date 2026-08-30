"""Tests for the K/V tensor quantizers + QuantizedFrame (Phase 3)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from membrane.quantization import (
    FORMAT_FP8_E4M3,
    FORMAT_FP8_E5M2,
    FORMAT_INT8,
    FORMAT_NF4,
    FP8E4M3Quantizer,
    FP8E5M2Quantizer,
    Int8PerChannelQuantizer,
    NF4Quantizer,
    QuantizedFrame,
    dequantize,
    quantize,
)


def _make_array(rows: int = 4, cols: int = 8, seed: int = 0) -> np.ndarray:
    """Build a small random float32 array for quantization tests."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)) * 10).astype("float32")


class TestInt8Quantizer:
    def test_round_trip_preserves_shape(self):
        q = Int8PerChannelQuantizer()
        tensor = _make_array()
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert out.shape == tensor.shape
        assert out.dtype == np.float32

    def test_zero_row_does_not_divide_by_zero(self):
        q = Int8PerChannelQuantizer()
        tensor = np.zeros((2, 4), dtype="float32")
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert np.allclose(out, tensor)

    def test_quantize_then_frame(self):
        tensor = _make_array()
        q = Int8PerChannelQuantizer()
        payload = q.quantize(tensor)
        frame = QuantizedFrame(
            format_id=FORMAT_INT8,
            scale=1.0,
            zero_point=0,
            original_dtype="float32",
            original_shape=tensor.shape,
            payload=payload,
        )
        decoded = dequantize(frame)
        assert decoded.shape == tensor.shape
        assert decoded.dtype == np.float32

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="bad magic"):
            QuantizedFrame.from_bytes(b"NOT-MVQF\x00\x00\x00\x00" + b"\x00" * 8)


class TestFP8E4M3Quantizer:
    def test_round_trip_preserves_shape(self):
        q = FP8E4M3Quantizer()
        tensor = _make_array()
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert out.shape == tensor.shape
        assert out.dtype == np.float32

    def test_zero_row_safe(self):
        q = FP8E4M3Quantizer()
        tensor = np.zeros((1, 8), dtype="float32")
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert np.allclose(out, tensor)


class TestFP8E5M2Quantizer:
    def test_round_trip_preserves_shape(self):
        q = FP8E5M2Quantizer()
        tensor = _make_array()
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert out.shape == tensor.shape
        assert out.dtype == np.float32


class TestNF4Quantizer:
    def test_round_trip_4bit_preserves_shape(self):
        q = NF4Quantizer()
        tensor = _make_array()
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert out.shape == tensor.shape
        assert out.dtype == np.float32

    def test_zero_row_safe(self):
        q = NF4Quantizer()
        tensor = np.zeros((1, 8), dtype="float32")
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert np.allclose(out, tensor)

    def test_odd_column_count(self):
        q = NF4Quantizer()
        tensor = _make_array(rows=2, cols=9)
        out = q.dequantize(
            q.quantize(tensor), "float32", tensor.shape
        )
        assert out.shape == tensor.shape

    def test_3d_original_shape_round_trip(self):
        q = NF4Quantizer()
        tensor = _make_array(rows=2, cols=8)
        out = q.dequantize(
            q.quantize(tensor), "float32", (2, 8, 1)
        )
        assert out.shape == (2, 8, 1)


class TestQuantizeFunction:
    def test_format_aliases(self):
        tensor = _make_array()
        for name, fmt in [
            ("int8", FORMAT_INT8),
            ("fp8_e4m3", FORMAT_FP8_E4M3),
            ("fp8_e5m2", FORMAT_FP8_E5M2),
            ("nf4", FORMAT_NF4),
        ]:
            frame = quantize(tensor, format_name=name)
            assert frame.format_id == fmt
            assert frame.original_shape == tensor.shape
            decoded = dequantize(frame)
            assert decoded.shape == tensor.shape

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown quantization"):
            quantize(_make_array(), format_name="ap4")

    def test_wire_round_trip(self):
        tensor = _make_array()
        frame = quantize(tensor, "int8")
        bytes_payload = frame.to_bytes()
        round_trip = QuantizedFrame.from_bytes(bytes_payload)
        assert round_trip.format_id == frame.format_id
        assert round_trip.scale == frame.scale
        assert round_trip.zero_point == frame.zero_point
        assert round_trip.original_dtype == frame.original_dtype
        assert round_trip.original_shape == frame.original_shape
        assert round_trip.payload == frame.payload

    def test_trailer_corruption_raises(self):
        tensor = _make_array()
        frame = quantize(tensor, "int8")
        bytes_payload = frame.to_bytes()
        corrupted = bytes_payload[:-1] + bytes([bytes_payload[-1] ^ 0x01])
        with pytest.raises(ValueError, match="trailer mismatch"):
            QuantizedFrame.from_bytes(corrupted)

    def test_short_payload_raises(self):
        tensor = _make_array()
        frame = quantize(tensor, "int8")
        with pytest.raises(ValueError, match="quantized frame too short"):
            QuantizedFrame.from_bytes(frame.to_bytes()[: 30])

    def test_dispatch_by_id(self):
        tensor = _make_array()
        for fmt in (FORMAT_INT8, FORMAT_FP8_E4M3, FORMAT_FP8_E5M2, FORMAT_NF4):
            frame = quantize(tensor, "int8")
            object.__setattr__(frame, "format_id", fmt)
            out = dequantize(frame)
            assert out.shape == tensor.shape
