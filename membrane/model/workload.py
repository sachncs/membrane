"""Workload generator for the case study in Section 4.1.

Requests follow a truncated log-normal distribution:
    mu = 9.90, sigma = 1.00, truncated to [128, 128K]
with mean uncached input length approximately 27K tokens.
Output length is fixed at 1024 tokens.
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

    Args:
        num_requests: Number of requests to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of integer token lengths.
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
        Tuple of (mean, p90).
    """
    if not values:
        return 0.0, 0.0
    sorted_values = sorted(values)
    mean = sum(sorted_values) / len(sorted_values)
    p90_index = int(math.ceil(0.9 * len(sorted_values))) - 1
    p90_index = max(0, min(p90_index, len(sorted_values) - 1))
    p90 = sorted_values[p90_index]
    return mean, p90


def conditional_means(
    lengths: List[int],
    threshold: int,
) -> tuple[float, float, float]:
    """Compute p = P(L > t), E[L | L > t], and E[L | L <= t].

    Args:
        lengths: List of request lengths.
        threshold: Routing threshold t.

    Returns:
        Tuple of (fraction_above, mean_long, mean_short).
    """
    longs = [l for l in lengths if l > threshold]
    shorts = [l for l in lengths if l <= threshold]

    p = len(longs) / len(lengths) if lengths else 0.0
    mean_long = sum(longs) / len(longs) if longs else 0.0
    mean_short = sum(shorts) / len(shorts) if shorts else 0.0

    return p, mean_long, mean_short
