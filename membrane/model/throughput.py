"""Analytical throughput model for the Membrane-PD architecture.

Implements Equations (1)–(6) from Section 3.4.1 of the paper
verbatim:

* Equation (1): per-instance KV throughput ``Phi_kv(l)``.
* Equation (3): Membrane stage throughput ``Theta_membrane``.
* Equation (4): PD-P stage throughput ``Theta_pd-p``.
* Equation (5): PD-D stage throughput ``Theta_pd-d``.
* Equation (6): end-to-end throughput ``Lambda_max``.

Each function delegates its core computation to
:mod:`membrane.model.profiler`, which provides the underlying
KV-size and prefill-time primitives.

References:
    * "Prefill-as-a-Service: KVCache of Next-Generation Models
      Could Go Cross-Datacenter", arXiv:2604.15039v2, §3.4.1.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.model import profiler


def kv_throughput(length: int) -> float:
    """Per-instance KV throughput ``Phi_kv(l)`` in Gbps.

    Equation (1):

    .. math::
        \\Phi_{kv}(l) = S_{kv}(l) / T_{prefill}(l)

    Args:
        length: Sequence length in tokens.

    Returns:
        float: KV throughput in Gbps.
    """
    return profiler.kv_throughput_gbps(length)


def stage_throughput_membrane(
    num_instances: int,
    egress_bandwidth_gbps: float,
    long_length: int,
    compute_scale: float = 1.0,
) -> float:
    """Membrane cluster throughput ``Theta_membrane`` in req/s.

    Equation (3):

    .. math::
        \\Theta_{membrane} = \\min(N_{membrane} / T_{prefill}(l_{long}),
                                    B_{out} / S_{kv}(l_{long}))

    Args:
        num_instances: Number of Membrane instances
            (``N_membrane``).
        egress_bandwidth_gbps: Membrane egress bandwidth in
            Gbps (``B_out``).
        long_length: Representative length for Membrane
            requests in tokens (``l_long``).
        compute_scale: Hardware compute scale for the Membrane
            cluster.

    Returns:
        float: Membrane throughput in requests per second.
    """
    compute_limit = num_instances / profiler.prefill_time_seconds(long_length, compute_scale)
    size_mib = profiler.kv_size_mib(long_length)
    # Convert Gbps to MiB/s: 1 Gbps = 1e9 bits/s = 1e9/8 bytes/s
    # = 1e9/8/1024/1024 MiB/s.
    bandwidth_mib_per_s = egress_bandwidth_gbps * 1e9 / 8.0 / (1024.0 * 1024.0)
    bandwidth_limit = bandwidth_mib_per_s / size_mib
    # The cluster is bottlenecked by whichever of compute or
    # bandwidth runs out first.
    return min(compute_limit, bandwidth_limit)


def stage_throughput_pd_p(
    num_instances: int,
    short_length: int,
    compute_scale: float = 1.0,
) -> float:
    """PD-P (local prefill) throughput ``Theta_pd-p`` in req/s.

    Equation (4):

    .. math::
        \\Theta_{pd-p} = N_p / T_{prefill}(l_{short})

    Args:
        num_instances: Number of PD-P instances (``N_p``).
        short_length: Representative length for PD-P requests
            in tokens (``l_short``).
        compute_scale: Hardware compute scale for the PD
            cluster.

    Returns:
        float: PD-P throughput in requests per second.
    """
    return num_instances / profiler.prefill_time_seconds(short_length, compute_scale)


def stage_throughput_pd_d(
    num_instances: int,
    max_batch_size: int,
    decode_time_seconds: float,
    output_length: int,
) -> float:
    """PD-D (decode) throughput ``Theta_pd-d`` in req/s.

    Equation (5):

    .. math::
        \\Theta_{pd-d} = (N_d \\cdot BS_{max}) / (T_{decode} \\cdot L_{out})

    Args:
        num_instances: Number of PD-D instances (``N_d``).
        max_batch_size: Maximum decode batch size
            (``BS_max``).
        decode_time_seconds: Per-step decode time in seconds
            (``T_decode``).
        output_length: Mean output length in tokens
            (``L_out``).

    Returns:
        float: PD-D throughput in requests per second.
    """
    return (num_instances * max_batch_size) / (decode_time_seconds * output_length)


def end_to_end_throughput(
    theta_membrane: float,
    theta_pd_p: float,
    theta_pd_d: float,
    fraction_to_membrane: float,
) -> float:
    """End-to-end system throughput ``Lambda_max`` in req/s.

    Equation (6):

    .. math::
        \\Lambda_{max} = \\min(\\Theta_{membrane} / p,
                                \\Theta_{pd-p} / (1 - p),
                                \\Theta_{pd-d})

    Args:
        theta_membrane: Membrane stage throughput in req/s.
        theta_pd_p: PD-P stage throughput in req/s.
        theta_pd_d: PD-D stage throughput in req/s.
        fraction_to_membrane: Fraction of requests routed to
            Membrane (``p``).

    Returns:
        float: End-to-end throughput in requests per second.
        ``+inf`` is reported for a stage whose fraction is
        outside ``(0, 1)`` so the ``min`` never picks it.
    """
    upstream_membrane = float("inf") if fraction_to_membrane <= 0.0 else theta_membrane / fraction_to_membrane

    upstream_pd_p = float("inf") if fraction_to_membrane >= 1.0 else theta_pd_p / (1.0 - fraction_to_membrane)

    # The system is bottlenecked by whichever stage has the
    # smallest scaled throughput.
    return min(upstream_membrane, upstream_pd_p, theta_pd_d)
