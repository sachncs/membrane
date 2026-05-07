"""Tests for the profiler module."""

import math

from membrane.model import profiler


def test_measured_points_exact():
    """KVCache sizes and latencies at measured points must match Table 5."""
    for length, expected_size, expected_time in zip(
        profiler.MEASURED_LENGTHS,
        profiler.MEASURED_KV_SIZES_MIB,
        profiler.MEASURED_PREFILL_TIMES,
    ):
        assert math.isclose(profiler.kv_size_mib(length), expected_size, rel_tol=1e-9)
        assert math.isclose(
            profiler.prefill_time_seconds(length), expected_time, rel_tol=1e-9
        )


def test_interpolation_between_points():
    """Interpolated values between measured points should lie between neighbors."""
    length = 2048  # between 1K and 8K
    size = profiler.kv_size_mib(length)
    time_s = profiler.prefill_time_seconds(length)
    assert 190.8 < size < 308.9
    assert 0.44 < time_s < 0.72


def test_clamping_below_minimum():
    """Values below the minimum measured length should be clamped."""
    assert profiler.kv_size_mib(64) == 190.8
    assert profiler.prefill_time_seconds(64) == 0.44


def test_clamping_above_maximum():
    """Values above the maximum measured length should be clamped."""
    assert profiler.kv_size_mib(200000) == 2316.3
    assert profiler.prefill_time_seconds(200000) == 7.40


def test_kv_throughput_consistency():
    """Computed KV throughput should approximately match Table 5 values."""
    for length, expected in zip(
        profiler.MEASURED_LENGTHS, profiler.MEASURED_KV_THROUGHPUTS
    ):
        computed = profiler.kv_throughput_gbps(length)
        # Allow 1% tolerance for unit-conversion rounding
        assert math.isclose(computed, expected, rel_tol=0.01)
