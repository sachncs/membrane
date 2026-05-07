"""Tests for the grid-search optimizer."""

import math

from membrane.model import optimizer, workload


def test_search_finds_valid_configuration():
    """Grid search must return a configuration within hardware limits."""
    lengths = workload.generate_request_lengths(10000, seed=42)
    t, n_p, n_d, lam = optimizer.search(lengths)
    assert optimizer.THRESHOLD_MIN <= t <= optimizer.THRESHOLD_MAX
    assert 1 <= n_p < optimizer.TOTAL_PD_INSTANCES
    assert n_d == optimizer.TOTAL_PD_INSTANCES - n_p
    assert lam > 0.0


def test_homogeneous_baseline_valid():
    """Homogeneous baseline must use all instances."""
    lengths = workload.generate_request_lengths(10000, seed=42)
    n_p, n_d, lam = optimizer.optimal_homogeneous_pd(lengths)
    total = optimizer.TOTAL_PD_INSTANCES + optimizer.MEMBRANE_INSTANCES
    assert n_p + n_d == total
    assert lam > 0.0


def test_naive_heterogeneous_valid():
    """Naive heterogeneous must return a positive throughput."""
    lengths = workload.generate_request_lengths(10000, seed=42)
    lam, _ = optimizer.naive_heterogeneous_pd(lengths)
    assert lam > 0.0


def test_membrane_pd_beats_homogeneous():
    """Membrane-PD should achieve higher throughput than homogeneous PD."""
    lengths = workload.generate_request_lengths(20000, seed=42)
    _, _, _, membrane_lambda = optimizer.search(lengths)
    _, _, hom_lambda = optimizer.optimal_homogeneous_pd(lengths)
    assert membrane_lambda > hom_lambda


def test_membrane_pd_beats_naive():
    """Membrane-PD should achieve higher throughput than naive heterogeneous."""
    lengths = workload.generate_request_lengths(20000, seed=42)
    _, _, _, membrane_lambda = optimizer.search(lengths)
    naive_lambda, _ = optimizer.naive_heterogeneous_pd(lengths)
    assert membrane_lambda > naive_lambda


def test_optimal_point_matches_paper():
    """Optimal configuration should be close to the paper's reported values.

    Paper reports:
        t ~ 19.4K, N_p=3, N_d=5, Lambda_max=3.24
    We allow tolerance for workload stochasticity and interpolation.
    """
    lengths = workload.generate_request_lengths(50000, seed=42)
    t, n_p, n_d, lam = optimizer.search(lengths)
    # Threshold should be in the vicinity of 19.4K
    assert 15000 <= t <= 25000
    # PD split should be close to 3/5
    assert n_p == 3 or n_p == 4
    assert n_d == 5 or n_d == 4
    # Lambda should be in the vicinity of 3.24 req/s
    assert 2.8 <= lam <= 3.6
