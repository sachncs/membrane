"""Throughput-optimal configuration via grid search.

Implements Section 3.4.2: given fixed hardware resources and
bandwidth, search over routing threshold ``t`` and the PD-cluster
prefill-to-decode ratio ``N_p / N_d`` to maximize ``Lambda_max``.

Three entry points are provided:

* :func:`evaluate_configuration` — evaluate ``Lambda_max`` for
  a single ``(t, N_p, N_d)`` triple.
* :func:`search` — exhaustive grid search over
  ``(t, N_p)``.
* :func:`optimal_homogeneous_pd` — grid search over ``(N_p,
  N_d)`` for a homogeneous PD baseline (no Membrane).
* :func:`naive_heterogeneous_pd` — evaluate the naive
  heterogeneous baseline where every request goes to Membrane.

References:
    * "Prefill-as-a-Service", arXiv:2604.15039v2, §3.4.2.

Assumptions (documented inline):

* H20 prefill is slower than H200. The single scale factor
  ``H20_COMPUTE_SCALE`` captures this difference uniformly
  because the paper does not provide per-length H20 profiles.
"""

import logging

logger = logging.getLogger(__name__)


from typing import List, Tuple

from membrane.model import throughput_model, workload

# Hardware constraints from Section 4.1.
TOTAL_PD_INSTANCES: int = 8  # 64 H20 GPUs / 8 GPUs per instance.
MEMBRANE_INSTANCES: int = 4  # 32 H200 GPUs / 8 GPUs per instance.
EGRESS_BANDWIDTH_GBPS: float = 100.0  # ~100 Gbps cross-cluster link.

# H20 prefill is slower than H200; the analytical H200 model
# is scaled by a single factor to approximate H20. See module
# docstring for the rationale.
H20_COMPUTE_SCALE: float = 2.5

# Decode constants inferred from Table 6.
DECODE_TIME_SECONDS: float = 0.025  # 25 ms per step.
MAX_BATCH_SIZE: int = 20
OUTPUT_LENGTH: int = 1024

# Grid search resolution.
THRESHOLD_MIN: int = 1024
THRESHOLD_MAX: int = 65536
THRESHOLD_STEP: int = 256


def evaluate_configuration(
    threshold: int,
    num_pd_p: int,
    num_pd_d: int,
    lengths: List[int],
) -> float:
    """Evaluate ``Lambda_max`` for a single configuration.

    Args:
        threshold: Routing threshold ``t`` in tokens.
        num_pd_p: Number of PD-P instances.
        num_pd_d: Number of PD-D instances.
        lengths: Workload lengths for computing conditional
            expectations.

    Returns:
        float: End-to-end throughput in req/s.
    """
    p, mean_long, mean_short = workload.conditional_means(lengths, threshold)

    # Fall back to ``threshold`` when a category is empty so
    # the throughput model has a well-defined representative
    # length to evaluate at.
    long_len = int(round(mean_long)) if mean_long > 0 else threshold
    short_len = int(round(mean_short)) if mean_short > 0 else threshold

    theta_membrane = throughput_model.stage_throughput_membrane(
        MEMBRANE_INSTANCES,
        EGRESS_BANDWIDTH_GBPS,
        long_len,
        compute_scale=1.0,
    )
    theta_pd_p = throughput_model.stage_throughput_pd_p(
        num_pd_p,
        short_len,
        compute_scale=H20_COMPUTE_SCALE,
    )
    theta_pd_d = throughput_model.stage_throughput_pd_d(
        num_pd_d,
        MAX_BATCH_SIZE,
        DECODE_TIME_SECONDS,
        OUTPUT_LENGTH,
    )

    return throughput_model.end_to_end_throughput(
        theta_membrane, theta_pd_p, theta_pd_d, p
    )


def search(
    lengths: List[int],
    total_pd_instances: int = TOTAL_PD_INSTANCES,
) -> Tuple[int, int, int, float]:
    """Grid search over ``t`` and ``N_p`` to maximize ``Lambda_max``.

    ``N_d`` is implicitly ``total_pd_instances - N_p``.

    Args:
        lengths: Workload lengths.
        total_pd_instances: Total number of PD instances
            available.

    Returns:
        tuple[int, int, int, float]: ``(optimal_threshold,
        optimal_n_p, optimal_n_d, optimal_lambda_max)``.
    """
    best_lambda = -1.0
    best_t = THRESHOLD_MIN
    best_n_p = 1
    best_n_d = total_pd_instances - 1

    for threshold in range(THRESHOLD_MIN, THRESHOLD_MAX + 1, THRESHOLD_STEP):
        for n_p in range(1, total_pd_instances):
            n_d = total_pd_instances - n_p
            lam = evaluate_configuration(threshold, n_p, n_d, lengths)
            if lam > best_lambda:
                best_lambda = lam
                best_t = threshold
                best_n_p = n_p
                best_n_d = n_d

    return best_t, best_n_p, best_n_d, best_lambda


def optimal_homogeneous_pd(
    lengths: List[int],
    total_instances: int = TOTAL_PD_INSTANCES + MEMBRANE_INSTANCES,
) -> Tuple[int, int, float]:
    """Optimize a homogeneous PD baseline (no Membrane).

    All instances are in one PD cluster. The function searches
    over the ``N_p / N_d`` ratio.

    Args:
        lengths: Workload lengths.
        total_instances: Total instances available (all H20
            equivalent).

    Returns:
        tuple[int, int, float]: ``(optimal_n_p, optimal_n_d,
        optimal_lambda_max)``.
    """
    best_lambda = -1.0
    best_n_p = 1
    best_n_d = total_instances - 1

    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    length = int(round(mean_length)) if mean_length > 0 else 32768

    for n_p in range(1, total_instances):
        n_d = total_instances - n_p
        theta_pd_p = throughput_model.stage_throughput_pd_p(
            n_p, length, compute_scale=H20_COMPUTE_SCALE
        )
        theta_pd_d = throughput_model.stage_throughput_pd_d(
            n_d, MAX_BATCH_SIZE, DECODE_TIME_SECONDS, OUTPUT_LENGTH
        )
        # Membrane is unreachable in this baseline, so its
        # contribution is +inf and the min() picks the PD
        # bottleneck.
        lam = throughput_model.end_to_end_throughput(
            float("inf"), theta_pd_p, theta_pd_d, 0.0
        )
        if lam > best_lambda:
            best_lambda = lam
            best_n_p = n_p
            best_n_d = n_d

    return best_n_p, best_n_d, best_lambda


def naive_heterogeneous_pd(
    lengths: List[int],
    membrane_instances: int = MEMBRANE_INSTANCES,
    pd_instances: int = TOTAL_PD_INSTANCES,
) -> Tuple[float, float]:
    """Evaluate naive heterogeneous PD (no selective routing).

    All prefill runs on Membrane (H200), all decode on PD (H20).
    There is no routing threshold; every request goes to
    Membrane.

    Args:
        lengths: Workload lengths.
        membrane_instances: Number of Membrane instances.
        pd_instances: Number of PD instances (all decode).

    Returns:
        tuple[float, float]: ``(lambda_max, mean_ttft_estimate)``.
        ``mean_ttft_estimate`` is always ``0.0`` because the
        paper does not model TTFT for this baseline.
    """
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    length = int(round(mean_length)) if mean_length > 0 else 32768

    theta_membrane = throughput_model.stage_throughput_membrane(
        membrane_instances, EGRESS_BANDWIDTH_GBPS, length
    )
    theta_pd_d = throughput_model.stage_throughput_pd_d(
        pd_instances, MAX_BATCH_SIZE, DECODE_TIME_SECONDS, OUTPUT_LENGTH
    )
    # fraction_to_membrane = 1.0 means PD-P's contribution is
    # +inf; only Membrane and PD-D bottlenecks apply.
    lam = throughput_model.end_to_end_throughput(
        theta_membrane, float("inf"), theta_pd_d, 1.0
    )
    return lam, 0.0
