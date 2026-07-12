"""CostModel: recompute vs reuse cost estimation.

This module defines :class:`CostModel`, a small but central
helper used by the routing and decision layers (notably
:class:`~membrane.offload_decision_engine.OffloadDecisionEngine`)
to decide whether reusing a cached KV segment is cheaper than
recomputing it on the local node.

The model has two inputs:

* **Compute scale** — multiplier applied to the analytical
  prefill time from
  :func:`membrane.model.profiler.prefill_time_seconds`. Higher
  values model faster hardware.
* **Bandwidth** — inter-node link bandwidth in Gbps used to
  estimate transfer latency.

The two derived quantities are:

* :meth:`precompute_cost_seconds` — time to recompute a prefix
  of ``n`` tokens on local hardware.
* :meth:`retrieval_cost_seconds` — time to transfer ``m`` MiB of
  cached KV across the network.

:meth:`reuse_is_cheaper` compares the two and lets callers pick
the cheaper option.

Limitations:
    * The model treats compute and retrieval as independent; it
      does not account for overlap, queueing, or backpressure.
    * Bandwidth is treated as a single shared link — multiple
      concurrent transfers contend for the same bandwidth.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.model.profiler import prefill_time_seconds


class CostModel:
    """Estimates whether reusing a cached KV is cheaper than recomputing.

    Attributes:
        compute_scale: Hardware compute scale factor applied
            to the analytical prefill time. ``1.0`` is the
            default hardware profile.
        bandwidth_gbps: Network bandwidth for KV transfer in
            Gbps.
    """

    def __init__(
        self,
        compute_scale: float = 1.0,
        bandwidth_gbps: float = 100.0,
    ) -> None:
        """Initialize the cost model.

        Args:
            compute_scale: Hardware compute scale factor.
                ``1.0`` represents the default hardware
                profile; values ``> 1`` model faster hardware.
            bandwidth_gbps: Network bandwidth in Gbps used
                for KV transfer latency estimation. Must be
                positive.
        """
        self.compute_scale = compute_scale
        self.bandwidth_gbps = bandwidth_gbps

    def precompute_cost_seconds(self, prefix_length: int) -> float:
        """Estimate the cost to precompute (prefill) a prefix.

        Args:
            prefix_length: Number of tokens to prefill.

        Returns:
            float: Estimated latency in seconds. Delegates to
            :func:`membrane.model.profiler.prefill_time_seconds`,
            which uses the analytical "Prefill-as-a-Service"
            throughput model.
        """
        return prefill_time_seconds(prefix_length, self.compute_scale)

    def retrieval_cost_seconds(self, kv_size_mib: float) -> float:
        """Estimate the cost to retrieve a cached KV across the network.

        Args:
            kv_size_mib: Size of the KV cache in MiB.

        Returns:
            float: Estimated transfer latency in seconds. A
            non-positive ``bandwidth_gbps`` returns ``+inf`` so
            that retrieval is always rejected in that case.
        """
        if self.bandwidth_gbps <= 0.0:
            return float("inf")
        # Convert MiB to Gbit: 1 MiB = 8.388608 Mb = 8.388608e-3 Gb.
        # Practical short form: size_mib * 8.388608.
        size_gbit = kv_size_mib * 8.388608
        return size_gbit / self.bandwidth_gbps

    def reuse_is_cheaper(
        self,
        prefix_length: int,
        kv_size_mib: float,
        retrieval_latency_seconds: float | None = None,
    ) -> bool:
        """Determine whether reuse (retrieval) is cheaper than recompute.

        Args:
            prefix_length: Tokens to prefill in the recompute
                path.
            kv_size_mib: KV cache size in the retrieval path.
            retrieval_latency_seconds: Optional override for
                the retrieval cost. When supplied, the
                bandwidth-based estimate is bypassed entirely —
                useful when the caller has a measured RTT for
                the specific link.

        Returns:
            bool: True if retrieval is cheaper than recompute,
            False otherwise.
        """
        compute_cost = self.precompute_cost_seconds(prefix_length)
        if retrieval_latency_seconds is not None:
            # Caller-supplied measurement overrides the
            # bandwidth-based estimate.
            retrieve_cost = retrieval_latency_seconds
        else:
            retrieve_cost = self.retrieval_cost_seconds(kv_size_mib)
        return retrieve_cost < compute_cost
