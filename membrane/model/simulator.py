"""Discrete-event simulator for end-to-end Membrane-PD evaluation.

The simulator runs a request stream through the routing and
scheduling pipeline and records stage-level throughput, TTFT,
and bandwidth utilization. It is intentionally simplified: it
does not model fine-grained GPU scheduling or TCP congestion,
but it does faithfully implement the analytical throughput
model and routing policies from the paper.

Three simulation scenarios are provided:

* :func:`run_membrane_pd` — Membrane-PD with the grid-search
  optimal configuration.
* :func:`run_homogeneous_pd` — homogeneous PD baseline (no
  Membrane).
* :func:`run_naive_heterogeneous_pd` — naive heterogeneous PD
  baseline where every request goes to Membrane.

References:
    * "Prefill-as-a-Service", arXiv:2604.15039v2, §4.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from membrane.model import metrics, optimizer, throughput_model, workload

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Aggregated results from a simulation run.

    Attributes:
        config_name: Human-readable label of the simulated
            configuration.
        lambda_max: End-to-end throughput in req/s.
        theta_membrane: Membrane stage throughput in req/s.
        theta_pd_p: PD-P stage throughput in req/s.
        theta_pd_d: PD-D stage throughput in req/s.
        threshold: Routing threshold ``t`` used in the run.
        num_membrane: Number of Membrane instances.
        num_pd_p: Number of PD-P instances.
        num_pd_d: Number of PD-D instances.
        mean_ttft: Mean time-to-first-token in seconds.
        p90_ttft: 90th-percentile TTFT in seconds.
        bandwidth_gbps: Egress bandwidth utilization in Gbps.
        fraction_to_membrane: Fraction of requests routed to
            Membrane.
        mean_long_length: Mean length of requests classified
            as long.
        mean_short_length: Mean length of requests classified
            as short.
    """

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
            Currently unused by the deterministic routing/TTFT
            logic but accepted for forward compatibility.

    Returns:
        SimulationResult: End-to-end metrics for the optimal
        Membrane-PD configuration.
    """
    # Optimize configuration.
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

    # Route every request to compute TTFT distribution.
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
        SimulationResult: End-to-end metrics for the optimal
        homogeneous PD configuration.
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

    # All requests go to PD-P.
    routed = [(length, "pd-p") for _unused in lengths]
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
        SimulationResult: End-to-end metrics for the naive
        heterogeneous PD configuration.
    """
    lam, _unused = optimizer.naive_heterogeneous_pd(lengths)

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

    # All requests go to Membrane.
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
