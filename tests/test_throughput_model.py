"""Tests for the analytical throughput model."""

import math

from membrane.model import throughput_model


def test_end_to_end_with_zero_fraction():
    """When p=0, Lambda_max = min(inf, Theta_pd-p / 1, Theta_pd-d)."""
    lam = throughput_model.end_to_end_throughput(10.0, 5.0, 3.0, 0.0)
    assert math.isclose(lam, 3.0)


def test_end_to_end_with_one_fraction():
    """When p=1, Lambda_max = min(Theta_membrane / 1, inf, Theta_pd-d)."""
    lam = throughput_model.end_to_end_throughput(10.0, 5.0, 3.0, 1.0)
    assert math.isclose(lam, 3.0)


def test_end_to_end_balanced():
    """When all stages are balanced, Lambda_max equals the common value."""
    lam = throughput_model.end_to_end_throughput(10.0, 10.0, 10.0, 0.5)
    # Theta_membrane/p = 10/0.5 = 20; Theta_pd-p/(1-p) = 10/0.5 = 20;
    # Theta_pd-d = 10. Lambda_max = min(20, 20, 10) = 10.
    assert math.isclose(lam, 10.0)


def test_stage_membrane_bandwidth_bound():
    """Membrane throughput should be bandwidth-bound when egress is tiny."""
    theta = throughput_model.stage_throughput_membrane(
        num_instances=100,
        egress_bandwidth_gbps=1.0,
        long_length=32768,
    )
    # With 1 Gbps and large KV, compute_limit is huge, so bandwidth dominates
    size_mib = 701.3
    bandwidth_mib_per_s = 1.0 * 1e9 / 8.0 / (1024.0 * 1024.0)
    expected = bandwidth_mib_per_s / size_mib
    assert math.isclose(theta, expected, rel_tol=1e-9)


def test_stage_membrane_compute_bound():
    """Membrane throughput should be compute-bound when bandwidth is huge."""
    theta = throughput_model.stage_throughput_membrane(
        num_instances=4,
        egress_bandwidth_gbps=1e6,
        long_length=32768,
    )
    expected = 4.0 / 1.84
    assert math.isclose(theta, expected, rel_tol=1e-9)


def test_pd_p_throughput():
    """PD-P throughput should equal N_p / T_prefill(l_short)."""
    theta = throughput_model.stage_throughput_pd_p(3, 8192)
    expected = 3.0 / 0.72
    assert math.isclose(theta, expected, rel_tol=1e-9)


def test_pd_d_throughput():
    """PD-D throughput should equal N_d * BS_max / (T_decode * L_out)."""
    theta = throughput_model.stage_throughput_pd_d(5, 20, 0.025, 1024)
    expected = (5.0 * 20.0) / (0.025 * 1024.0)
    assert math.isclose(theta, expected, rel_tol=1e-9)
