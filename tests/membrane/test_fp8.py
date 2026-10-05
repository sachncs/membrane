"""fp8 quantization: correct code points, nearest rounding, no sign corruption."""

import numpy as np
import pytest

from membrane.quantization import FP8E4M3Quantizer, FP8E5M2Quantizer, dequantize, quantize


def test_codebooks_match_the_formats() -> None:
    e4m3, e5m2 = FP8E4M3Quantizer.codebook, FP8E5M2Quantizer.codebook
    assert e4m3.max_value == 448.0 and e5m2.max_value == 57344.0
    assert 0.0 in e4m3.sorted_values and 1.0 in e4m3.sorted_values and 1.125 in e4m3.sorted_values
    assert 1.25 in e5m2.sorted_values and 1.125 not in e5m2.sorted_values  # 2 mantissa bits
    assert np.isnan(e4m3.decode[0x7F]) and e4m3.decode[0x7E] == 448.0


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_large_magnitudes_keep_their_sign(fmt: str) -> None:
    # The old int8 fallback wrapped values above 127 (after scaling) and flipped signs.
    tensor = np.linspace(-8.0, 8.0, 512, dtype="float32").reshape(4, 128)
    restored = dequantize(quantize(tensor, fmt))
    big = np.abs(tensor) > 0.5
    assert not np.any(np.sign(restored[big]) != np.sign(tensor[big]))
    relative = np.abs(restored[big] - tensor[big]) / np.abs(tensor[big])
    assert relative.max() < (0.07 if fmt == "fp8_e4m3" else 0.13)


def test_exact_code_points_round_trip_exactly() -> None:
    row = np.array([[448.0, -448.0, 1.0, -0.5, 0.0, 240.0]], dtype="float32")
    assert np.array_equal(dequantize(quantize(row, "fp8_e4m3")), row)


def test_one_byte_per_element() -> None:
    tensor = np.ones((8, 256), dtype="float16")
    frame = quantize(tensor, "fp8_e4m3")
    assert len(frame.payload) == 8 + 8 * 4 + 8 * 256
