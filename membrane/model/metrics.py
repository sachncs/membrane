"""Evaluation metrics for the Membrane-PD case study.

Metrics derived from Section 4 of the paper:

* ``Lambda_max`` — sustainable system throughput (req/s).
* Mean TTFT and P90 TTFT — latency in seconds.
* Cross-datacenter bandwidth utilization — Gbps.

The module exposes three top-level functions:

* :func:`compute_ttft` — TTFT for a single request.
* :func:`aggregate_ttft` — mean and P90 over a batch.
* :func:`bandwidth_utilization` — egress bandwidth demand.

References:
    * "Prefill-as-a-Service", arXiv:2604.15039v2, §4.
"""

import logging

logger = logging.getLogger(__name__)


import math
from typing import List, Tuple

from membrane.model import profiler


def compute_ttft(
    length: int,
    target: str,
    membrane_bandwidth_gbps: float = 100.0,
    compute_scale: float = 1.0,
) -> float:
    """Estimate Time-To-First-Token for a single request.

    * For PD-P requests: ``TTFT = T_prefill(l)``.
    * For Membrane requests: ``TTFT = T_prefill(l) + KV transfer time``,
      where ``KV transfer time = S_kv(l) / bandwidth``.

    Args:
        length: Uncached input length in tokens.
        target: ``"membrane"`` or ``"pd-p"``.
        membrane_bandwidth_gbps: Available cross-cluster
            bandwidth in Gbps.
        compute_scale: Hardware compute scale for prefill
            latency.

    Returns:
        float: Estimated TTFT in seconds.
    """
    prefill = profiler.prefill_time_seconds(length, compute_scale)
    if target == "pd-p":
        return prefill

    size_mib = profiler.kv_size_mib(length)
    # MiB -> Gbit: size_mib * 1024 * 1024 * 8 / 1e9.
    size_gbit = size_mib * 1024.0 * 1024.0 * 8.0 / 1e9
    transfer_time = size_gbit / membrane_bandwidth_gbps
    return prefill + transfer_time


def aggregate_ttft(
    lengths_and_targets: List[Tuple[int, str]],
    membrane_bandwidth_gbps: float = 100.0,
    pd_compute_scale: float = 1.0,
) -> Tuple[float, float]:
    """Compute mean and P90 TTFT over a batch of requests.

    Args:
        lengths_and_targets: List of ``(length, target)``
            pairs.
        membrane_bandwidth_gbps: Cross-cluster bandwidth in
            Gbps.
        pd_compute_scale: Hardware compute scale for PD-P
            prefill.

    Returns:
        tuple[float, float]: ``(mean_ttft, p90_ttft)`` in
        seconds. Both are ``0.0`` when the input list is empty.
    """
    ttfts = [
        compute_ttft(
            length,
            target,
            membrane_bandwidth_gbps,
            compute_scale=pd_compute_scale if target == "pd-p" else 1.0,
        )
        for length, target in lengths_and_targets
    ]
    if not ttfts:
        return 0.0, 0.0

    sorted_ttfts = sorted(ttfts)
    mean = sum(sorted_ttfts) / len(sorted_ttfts)
    # 0-based index of the 90th percentile; clamped to a valid
    # index for very small batches.
    p90_index = int(math.ceil(0.9 * len(sorted_ttfts))) - 1
    p90_index = max(0, min(p90_index, len(sorted_ttfts) - 1))
    p90 = sorted_ttfts[p90_index]
    return mean, p90


def bandwidth_utilization(
    lambda_rate: float,
    fraction_to_membrane: float,
    mean_long_length: int,
    membrane_instances: int = 4,
) -> float:
    """Estimate average Membrane egress bandwidth utilization in Gbps.

    The function computes the aggregate egress load
    ``lambda_rate * p * S_kv(l_long) / T_prefill(l_long)`` and
    converts it from MiB/s to Gbps.

    Args:
        lambda_rate: System throughput in req/s.
        fraction_to_membrane: Fraction ``p`` routed to
            Membrane.
        mean_long_length: Mean length of Membrane requests in
            tokens.
        membrane_instances: Number of Membrane instances.
            Currently unused but accepted for forward
            compatibility with utilization-vs-capacity
            derivations.

    Returns:
        float: Estimated average egress bandwidth in Gbps.
        ``0.0`` when ``fraction_to_membrane <= 0``.
    """
    if fraction_to_membrane <= 0.0:
        return 0.0

    size_mib = profiler.kv_size_mib(mean_long_length)
    time_s = profiler.prefill_time_seconds(mean_long_length)
    # MiB per second for all offloaded requests.
    mib_per_s = lambda_rate * fraction_to_membrane * size_mib / time_s
    # Convert to Gbps: 1 MiB/s = 1024*1024*8/1e9 Gbps.
    gbps = mib_per_s * 1024.0 * 1024.0 * 8.0 / 1e9
    return gbps
