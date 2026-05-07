"""Discrete-event simulator for end-to-end Membrane-PD evaluation.

The simulator runs a request stream through the routing and scheduling
pipeline and records stage-level throughput, TTFT, and bandwidth
utilization. It is intentionally simplified: it does not model fine-grained
GPU scheduling or TCP congestion, but it does faithfully implement the
analytical throughput model and routing policies from the paper.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from membrane.model import metrics, optimizer, throughput_model, workload

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Aggregated results from a simulation run."""

    config_name: str
    lambda_max: float
    theta_membrane: float
    theta_pd_p: float
    theta_pd_d: float
    threshold: int
    num_membrane: int
    num_pd_p: int
    num_pd_d: int
    mean_ttft: float
    p90_ttft: float
    bandwidth_gbps: float
    fraction_to_membrane: float
    mean_long_length: float
    mean_short_length: float


def run_membrane_pd(
    lengths: List[int],
    seed: int = 42,
) -> SimulationResult:
    """Simulate the Membrane-PD architecture with optimal configuration.

    Args:
        lengths: Workload request lengths.
        seed: Random seed for any stochastic sub-components.

    Returns:
        SimulationResult with all metrics.
    """
    # Optimize configuration
    best_t, best_n_p, best_n_d, best_lambda = optimizer.search(lengths)

    p, mean_long, mean_short = workload.conditional_means(lengths, best_t)

    long_len = int(round(mean_long)) if mean_long > 0 else best_t
    short_len = int(round(mean_short)) if mean_short > 0 else best_t

    theta_membrane = throughput_model.stage_throughput_membrane(
        optimizer.MEMBRANE_INSTANCES,
        optimizer.EGRESS_BANDWIDTH_GBPS,
        long_len,
        compute_scale=1.0,
    )
    theta_pd_p = throughput_model.stage_throughput_pd_p(
        best_n_p, short_len, compute_scale=optimizer.H20_COMPUTE_SCALE
    )
    theta_pd_d = throughput_model.stage_throughput_pd_d(
        best_n_d,
        optimizer.MAX_BATCH_SIZE,
        optimizer.DECODE_TIME_SECONDS,
        optimizer.OUTPUT_LENGTH,
    )

    # Route every request to compute TTFT distribution
    from membrane.model import router

    rtr = router.Router(best_t, bandwidth_abundant=False)
    routed: List[Tuple[int, str]] = []
    for total_len in lengths:
        decision = rtr.route(total_len, cached_prefix_membrane=0, cached_prefix_pd=0)
        routed.append((decision.incremental_length, decision.target))

    mean_ttft, p90_ttft = metrics.aggregate_ttft(
        routed, pd_compute_scale=optimizer.H20_COMPUTE_SCALE
    )
    bw = metrics.bandwidth_utilization(
        best_lambda, p, long_len, optimizer.MEMBRANE_INSTANCES
    )

    return SimulationResult(
        config_name="Membrane-PD",
        lambda_max=best_lambda,
        theta_membrane=theta_membrane,
        theta_pd_p=theta_pd_p,
        theta_pd_d=theta_pd_d,
        threshold=best_t,
        num_membrane=optimizer.MEMBRANE_INSTANCES,
        num_pd_p=best_n_p,
        num_pd_d=best_n_d,
        mean_ttft=mean_ttft,
        p90_ttft=p90_ttft,
        bandwidth_gbps=bw,
        fraction_to_membrane=p,
        mean_long_length=mean_long,
        mean_short_length=mean_short,
    )


def run_homogeneous_pd(
    lengths: List[int],
) -> SimulationResult:
    """Simulate the homogeneous PD baseline.

    Args:
        lengths: Workload request lengths.

    Returns:
        SimulationResult with all metrics.
    """
    total_instances = optimizer.TOTAL_PD_INSTANCES + optimizer.MEMBRANE_INSTANCES
    best_n_p, best_n_d, best_lambda = optimizer.optimal_homogeneous_pd(
        lengths, total_instances
    )

    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    length = int(round(mean_length)) if mean_length > 0 else 32768

    theta_pd_p = throughput_model.stage_throughput_pd_p(
        best_n_p, length, compute_scale=optimizer.H20_COMPUTE_SCALE
    )
    theta_pd_d = throughput_model.stage_throughput_pd_d(
        best_n_d,
        optimizer.MAX_BATCH_SIZE,
        optimizer.DECODE_TIME_SECONDS,
        optimizer.OUTPUT_LENGTH,
    )

    # All requests go to PD-P
    routed = [(length, "pd-p") for unused in lengths]
    mean_ttft, p90_ttft = metrics.aggregate_ttft(
        routed, pd_compute_scale=optimizer.H20_COMPUTE_SCALE
    )

    return SimulationResult(
        config_name="Homogeneous PD",
        lambda_max=best_lambda,
        theta_membrane=0.0,
        theta_pd_p=theta_pd_p,
        theta_pd_d=theta_pd_d,
        threshold=0,
        num_membrane=0,
        num_pd_p=best_n_p,
        num_pd_d=best_n_d,
        mean_ttft=mean_ttft,
        p90_ttft=p90_ttft,
        bandwidth_gbps=0.0,
        fraction_to_membrane=0.0,
        mean_long_length=mean_length,
        mean_short_length=mean_length,
    )


def run_naive_heterogeneous_pd(
    lengths: List[int],
) -> SimulationResult:
    """Simulate the naive heterogeneous PD baseline.

    Args:
        lengths: Workload request lengths.

    Returns:
        SimulationResult with all metrics.
    """
    lam, unused = optimizer.naive_heterogeneous_pd(lengths)

    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    length = int(round(mean_length)) if mean_length > 0 else 32768

    theta_membrane = throughput_model.stage_throughput_membrane(
        optimizer.MEMBRANE_INSTANCES,
        optimizer.EGRESS_BANDWIDTH_GBPS,
        length,
    )
    theta_pd_d = throughput_model.stage_throughput_pd_d(
        optimizer.TOTAL_PD_INSTANCES,
        optimizer.MAX_BATCH_SIZE,
        optimizer.DECODE_TIME_SECONDS,
        optimizer.OUTPUT_LENGTH,
    )

    # All requests go to Membrane
    routed = [(length, "membrane") for _ in lengths]
    mean_ttft, p90_ttft = metrics.aggregate_ttft(routed)

    return SimulationResult(
        config_name="Naive Heterogeneous PD",
        lambda_max=lam,
        theta_membrane=theta_membrane,
        theta_pd_p=0.0,
        theta_pd_d=theta_pd_d,
        threshold=0,
        num_membrane=optimizer.MEMBRANE_INSTANCES,
        num_pd_p=0,
        num_pd_d=optimizer.TOTAL_PD_INSTANCES,
        mean_ttft=mean_ttft,
        p90_ttft=p90_ttft,
        bandwidth_gbps=metrics.bandwidth_utilization(
            lam, 1.0, length, optimizer.MEMBRANE_INSTANCES
        ),
        fraction_to_membrane=1.0,
        mean_long_length=mean_length,
        mean_short_length=0.0,
    )
