"""Tests for the workload generator."""

import math

from membrane.model import workload


def test_truncation_bounds():
    """All generated lengths must lie within [128, 128K]."""
    lengths = workload.generate_request_lengths(10000, seed=123)
    assert all(workload.MIN_LENGTH <= l <= workload.MAX_LENGTH for l in lengths)


def test_mean_approximate():
    """Mean of a large sample should be approximately 27K tokens."""
    lengths = workload.generate_request_lengths(50000, seed=42)
    mean = sum(lengths) / len(lengths)
    # Paper says "mean of approximately 27K tokens"
    assert 25000 < mean < 29000


def test_output_length_constant():
    """Output length is fixed at 1024 tokens."""
    assert workload.OUTPUT_LENGTH == 1024


def test_conditional_means():
    """Conditional means should partition the data correctly."""
    lengths = [100, 500, 1000, 5000, 10000, 50000]
    p, mean_long, mean_short = workload.conditional_means(lengths, threshold=1000)
    assert math.isclose(p, 3 / 6)
    assert math.isclose(mean_long, (5000 + 10000 + 50000) / 3)
    assert math.isclose(mean_short, (100 + 500 + 1000) / 3)


def test_mean_and_p90():
    """Mean and P90 should match hand-calculated values."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    mean, p90 = workload.mean_and_p90(values)
    assert math.isclose(mean, 5.5)
    assert math.isclose(p90, 9.0)
