"""Tests for the end-to-end simulator."""

import math

from membrane.model import simulator, workload


def test_simulation_result_fields():
    """SimulationResult must contain all expected fields."""
    lengths = workload.generate_request_lengths(1000, seed=1)
    result = simulator.run_membrane_pd(lengths)
    assert result.config_name == "Membrane-PD"
    assert result.lambda_max > 0.0
    assert result.threshold > 0
    assert result.num_membrane > 0
    assert result.num_pd_p > 0
    assert result.num_pd_d > 0
    assert result.mean_ttft > 0.0
    assert result.p90_ttft >= result.mean_ttft


def test_homogeneous_vs_membrane_ttft():
    """Membrane-PD should have lower mean TTFT than homogeneous PD."""
    lengths = workload.generate_request_lengths(5000, seed=2)
    result = simulator.run_membrane_pd(lengths)
    hom = simulator.run_homogeneous_pd(lengths)
    assert result.mean_ttft < hom.mean_ttft


def test_bandwidth_utilization_within_budget():
    """Membrane egress bandwidth should stay well below 100 Gbps."""
    lengths = workload.generate_request_lengths(5000, seed=3)
    result = simulator.run_membrane_pd(lengths)
    assert result.bandwidth_gbps < 100.0
    # Paper reports ~13 Gbps average utilization. With small samples
    # the exact value varies; we just check it is modest (< 30 Gbps).
    assert 0.0 <= result.bandwidth_gbps <= 30.0


def test_naive_has_higher_bandwidth():
    """Naive heterogeneous should consume more bandwidth than Membrane-PD."""
    lengths = workload.generate_request_lengths(5000, seed=4)
    result = simulator.run_membrane_pd(lengths)
    naive = simulator.run_naive_heterogeneous_pd(lengths)
    assert naive.bandwidth_gbps > result.bandwidth_gbps


def test_fraction_to_membrane_reasonable():
    """Fraction offloaded should be near the paper's reported ~50%."""
    lengths = workload.generate_request_lengths(20000, seed=5)
    result = simulator.run_membrane_pd(lengths)
    assert 0.3 <= result.fraction_to_membrane <= 0.7
