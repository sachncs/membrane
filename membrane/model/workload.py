"""Workload generator for the case study in Section 4.1.

Requests follow a truncated log-normal distribution:

* ``mu = 9.90``
* ``sigma = 1.00``
* Truncated to ``[128, 128K]``

The resulting distribution has a mean uncached input length of
roughly 27K tokens. Output length is fixed at 1024 tokens.

The module exposes:

* :func:`generate_request_lengths` — sample a workload.
* :func:`mean_and_p90` — compute summary statistics.
* :func:`conditional_means` — compute the routing-relevant
  conditional expectations for a given threshold.

References:
    * "Prefill-as-a-Service", arXiv:2604.15039v2, §4.1.
"""

import logging

logger = logging.getLogger(__name__)


import math
import random
from typing import List

MU: float = 9.90
SIGMA: float = 1.00
MIN_LENGTH: int = 128
MAX_LENGTH: int = 131072
OUTPUT_LENGTH: int = 1024


def generate_request_lengths(
    num_requests: int,
    seed: int = 42,
) -> List[int]:
    """Generate a list of uncached input lengths.

    Sampling is performed via :func:`random.Random.lognormvariate`
    with ``MU`` and ``SIGMA``. Samples outside ``[MIN_LENGTH,
    MAX_LENGTH]`` are rejected and the draw is retried, so the
    resulting sequence is exactly the truncated distribution
    described in the paper.

    Args:
        num_requests: Number of requests to generate.
        seed: Random seed for reproducibility.

    Returns:
        list[int]: ``num_requests`` integer token lengths,
        each in ``[MIN_LENGTH, MAX_LENGTH]``.
    """
    rng = random.Random(seed)
    lengths: List[int] = []
    while len(lengths) < num_requests:
        sample = rng.lognormvariate(MU, SIGMA)
        length = int(round(sample))
        if MIN_LENGTH <= length <= MAX_LENGTH:
            lengths.append(length)
    return lengths


def mean_and_p90(values: List[float]) -> tuple[float, float]:
    """Return mean and 90th percentile of a list of values.

    Args:
        values: List of numeric values.

    Returns:
        tuple[float, float]: ``(mean, p90)``. Both are ``0.0``
        when ``values`` is empty.
    """
    if not values:
        return 0.0, 0.0
    sorted_values = sorted(values)
    mean = sum(sorted_values) / len(sorted_values)
    # 0-based index of the 90th percentile. Clamp to a valid
    # index so very small lists return the largest element
    # rather than raising.
    p90_index = int(math.ceil(0.9 * len(sorted_values))) - 1
    p90_index = max(0, min(p90_index, len(sorted_values) - 1))
    p90 = sorted_values[p90_index]
    return mean, p90


def conditional_means(
    lengths: List[int],
    threshold: int,
) -> tuple[float, float, float]:
    """Compute ``P(L > t)``, ``E[L | L > t]``, and ``E[L | L <= t]``.

    These three conditional statistics are the inputs the
    routing layer needs to estimate the long/short split for a
    given threshold.

    Args:
        lengths: List of request lengths.
        threshold: Routing threshold ``t``.

    Returns:
        tuple[float, float, float]: ``(fraction_above,
        mean_long, mean_short)``. Means are ``0.0`` when the
        corresponding group is empty.
    """
    longs = [l for l in lengths if l > threshold]
    shorts = [l for l in lengths if l <= threshold]

    p = len(longs) / len(lengths) if lengths else 0.0
    mean_long = sum(longs) / len(longs) if longs else 0.0
    mean_short = sum(shorts) / len(shorts) if shorts else 0.0

    return p, mean_long, mean_short
