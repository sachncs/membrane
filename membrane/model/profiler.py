"""Model profiler that provides ``S_kv(l)`` and ``T_prefill(l)`` for the internal 1T hybrid model.

Data is taken from Table 5 of the paper. Values between measured
points are linearly interpolated. Extrapolation beyond the
measured range is clamped to the nearest measured value and
should be treated as an approximation.

References:
    * "Prefill-as-a-Service: KVCache of Next-Generation Models
      Could Go Cross-Datacenter", arXiv:2604.15039v2, Table 5.

The module exposes three public estimators:

* :func:`kv_size_mib` — KV cache size in MiB.
* :func:`prefill_time_seconds` — prefill latency in seconds.
* :func:`kv_throughput_gbps` — per-instance KV throughput in
  Gbps, derived from the other two.

The internal :func:`interpolate` helper performs the piecewise
linear interpolation with boundary clamping.
"""

import logging

logger = logging.getLogger(__name__)


from typing import Sequence

# Measured sequence lengths (tokens), Table 5.
MEASURED_LENGTHS: Sequence[int] = (1024, 8192, 32768, 131072)

# Measured KVCache sizes (MiB), Table 5.
MEASURED_KV_SIZES_MIB: Sequence[float] = (190.8, 308.9, 701.3, 2316.3)

# Measured prefill latencies (seconds), Table 5.
MEASURED_PREFILL_TIMES: Sequence[float] = (0.44, 0.72, 1.84, 7.40)

# Measured KV throughputs (Gbps), Table 5 (cross-check only).
MEASURED_KV_THROUGHPUTS: Sequence[float] = (3.61, 3.59, 3.19, 2.62)


def kv_size_mib(length: int) -> float:
    """Return interpolated KVCache size in MiB for ``length``.

    Args:
        length: Sequence length in tokens.

    Returns:
        float: KVCache size in MiB. Clamped to the nearest
        measured value when ``length`` falls outside the
        measured range.
    """
    return interpolate(length, MEASURED_LENGTHS, MEASURED_KV_SIZES_MIB)


def prefill_time_seconds(length: int, compute_scale: float = 1.0) -> float:
    """Return interpolated prefill latency for ``length``.

    Args:
        length: Sequence length in tokens.
        compute_scale: Multiplicative factor for prefill
            latency to account for hardware differences
            (e.g., H20 vs H200). Default ``1.0`` corresponds
            to the H200 profile from Table 5.

    Returns:
        float: Prefill latency in seconds, multiplied by
        ``compute_scale``.
    """
    base = interpolate(length, MEASURED_LENGTHS, MEASURED_PREFILL_TIMES)
    return base * compute_scale


def kv_throughput_gbps(length: int) -> float:
    """Return per-instance KV throughput in Gbps for ``length``.

    Computed as ``S_kv(l) / T_prefill(l)`` and converted from
    MiB/s to Gbps. The conversion uses the standard
    binary-to-decimal approximation consistent with the paper's
    reported throughput values.

    Args:
        length: Sequence length in tokens.

    Returns:
        float: KV throughput in Gbps.
    """
    size_mib = kv_size_mib(length)
    time_s = prefill_time_seconds(length)
    # MiB -> bits: size_mib * 1024 * 1024 * 8.
    # bits/s -> Gbps: / 1e9.
    gbps = (size_mib * 1024.0 * 1024.0 * 8.0) / (time_s * 1e9)
    return gbps


def interpolate(x: int, xs: Sequence[int], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation with clamping at boundaries.

    For ``x`` strictly within the ``xs`` range, the returned
    value is the linear interpolation between the surrounding
    measured points. For ``x`` outside the range the function
    returns the nearest measured value — extrapolation is not
    performed.

    Args:
        x: Query point.
        xs: Known x coordinates (must be strictly increasing).
        ys: Known y coordinates.

    Returns:
        float: Interpolated or clamped y value.

    Raises:
        ValueError: When ``xs`` is empty.
    """
    if not xs:
        raise ValueError("interpolate requires at least one data point")
    if x <= xs[0]:
        # Below the lowest measured point — clamp.
        return ys[0]
    if x >= xs[-1]:
        # Above the highest measured point — clamp.
        return ys[-1]

    # Linear interpolation between the two surrounding
    # measured points.
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            y0, y1 = ys[i], ys[i + 1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    # Defensive fallback: the boundary clamps above mean we
    # never reach here. Returning ys[-1] keeps the function
    # total in the face of unexpected floating-point corner
    # cases.
    return ys[-1]
