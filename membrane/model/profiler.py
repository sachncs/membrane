"""Model profiler that provides S_kv(l) and T_prefill(l) for the internal 1T hybrid model.

Data is taken from Table 5 of the paper. Values between measured points are
linearly interpolated. Extrapolation beyond the measured range is clamped to
the nearest measured value and flagged as an approximation.
"""

import logging

logger = logging.getLogger(__name__)


from typing import Sequence

# Measured sequence lengths (tokens)
MEASURED_LENGTHS: Sequence[int] = (1024, 8192, 32768, 131072)

# Measured KVCache sizes (MiB) from Table 5
MEASURED_KV_SIZES_MIB: Sequence[float] = (190.8, 308.9, 701.3, 2316.3)

# Measured prefill latencies (seconds) from Table 5
MEASURED_PREFILL_TIMES: Sequence[float] = (0.44, 0.72, 1.84, 7.40)

# Measured KV throughputs (Gbps) from Table 5 (for cross-check only)
MEASURED_KV_THROUGHPUTS: Sequence[float] = (3.61, 3.59, 3.19, 2.62)


def kv_size_mib(length: int) -> float:
    """Return interpolated KVCache size in MiB for a given sequence length.

    Args:
        length: Sequence length in tokens.

    Returns:
        KVCache size in MiB.
    """
    return interpolate(length, MEASURED_LENGTHS, MEASURED_KV_SIZES_MIB)


def prefill_time_seconds(length: int, compute_scale: float = 1.0) -> float:
    """Return interpolated prefill latency in seconds for a given sequence length.

    Args:
        length: Sequence length in tokens.
        compute_scale: Multiplicative factor for prefill latency to account
            for hardware differences (e.g., H20 vs H200). Default is 1.0
            (H200 profile from Table 5).

    Returns:
        Prefill latency in seconds.
    """
    base = interpolate(length, MEASURED_LENGTHS, MEASURED_PREFILL_TIMES)
    return base * compute_scale


def kv_throughput_gbps(length: int) -> float:
    """Return per-instance KV throughput in Gbps for a given sequence length.

    This is computed as S_kv(l) / T_prefill(l) and converted to Gbps
    (1 MiB = 8 Mib, 1 Mib/s = 1.048576 Mbit/s; we use the standard
    binary-to-decimal approximation consistent with the paper's reported
    values).

    Args:
        length: Sequence length in tokens.

    Returns:
        KV throughput in Gbps.
    """
    size_mib = kv_size_mib(length)
    time_s = prefill_time_seconds(length)
    # MiB -> bits: size_mib * 1024 * 1024 * 8
    # bits/s -> Gbps: / (1000 * 1000 * 1000)
    gbps = (size_mib * 1024.0 * 1024.0 * 8.0) / (time_s * 1e9)
    return gbps


def interpolate(x: int, xs: Sequence[int], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation with clamping at boundaries.

    Args:
        x: Query point.
        xs: Known x coordinates (must be strictly increasing).
        ys: Known y coordinates.

    Returns:
        Interpolated or clamped y value.
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            y0, y1 = ys[i], ys[i + 1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    # Should never reach here because of the clamp checks above.
    return ys[-1]
