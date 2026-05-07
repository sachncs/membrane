"""CostModel: recompute vs reuse cost estimation."""

import logging

logger = logging.getLogger(__name__)


from membrane.model.profiler import prefill_time_seconds


class CostModel:
    """Estimates whether reusing a cached KV is cheaper than recomputing."""

    def __init__(
        self,
        compute_scale: float = 1.0,
        bandwidth_gbps: float = 100.0,
    ) -> None:
        """Initialize the cost model.

        Args:
            compute_scale: Hardware compute scale factor.
            bandwidth_gbps: Network bandwidth for KV transfer.
        """
        self.compute_scale = compute_scale
        self.bandwidth_gbps = bandwidth_gbps

    def precompute_cost_seconds(self, prefix_length: int) -> float:
        """Estimated cost to precompute (prefill) a prefix.

        Args:
            prefix_length: Number of tokens to prefill.

        Returns:
            Estimated latency in seconds.
        """
        return prefill_time_seconds(prefix_length, self.compute_scale)

    def retrieval_cost_seconds(self, kv_size_mib: float) -> float:
        """Estimated cost to retrieve a cached KV across the network.

        Args:
            kv_size_mib: Size of the KV cache in MiB.

        Returns:
            Estimated transfer latency in seconds.
        """
        if self.bandwidth_gbps <= 0.0:
            return float("inf")
        # MiB -> Gb: size * 8 * 1024 * 1024 / 1e9 = size * 8.388608
        size_gbit = kv_size_mib * 8.388608
        return size_gbit / self.bandwidth_gbps

    def reuse_is_cheaper(
        self,
        prefix_length: int,
        kv_size_mib: float,
        retrieval_latency_seconds: float | None = None,
    ) -> bool:
        """Determine if reuse (retrieval) is cheaper than recompute.

        Args:
            prefix_length: Tokens to prefill.
            kv_size_mib: KV cache size for retrieval cost.
            retrieval_latency_seconds: Optional override for retrieval cost.

        Returns:
            True if retrieval is cheaper than recompute.
        """
        compute_cost = self.precompute_cost_seconds(prefix_length)
        if retrieval_latency_seconds is not None:
            retrieve_cost = retrieval_latency_seconds
        else:
            retrieve_cost = self.retrieval_cost_seconds(kv_size_mib)
        return retrieve_cost < compute_cost
