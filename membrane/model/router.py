"""Request router implementing the routing policies in Section 3.3 and 3.4.3.

The router decides whether a request is handled locally (PD-P) or offloaded
to a Membrane cluster based on its incremental uncached length and current
bandwidth conditions.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    """Outcome of a routing decision."""

    target: str  # "membrane" or "pd-p"
    incremental_length: int
    cached_prefix_length: int
    cross_cluster_cache_transfer: bool = False


class Router:
    """Length-based threshold router with optional prefix-cache awareness.

    Implements the policies described in:
      - Section 3.3: "When l > t, the request is routed to a Membrane cluster"
      - Section 3.4.3 short-term: bandwidth-scarce vs bandwidth-abundant
        prefix-cache routing.
    """

    def __init__(
        self,
        threshold: int,
        bandwidth_abundant: bool = False,
    ):
        """Initialize the router.

        Args:
            threshold: Routing threshold t in tokens.
            bandwidth_abundant: If True, consider global best cache and allow
                cross-cluster cache transfer. If False, evaluate caches
                independently per cluster.
        """
        self.threshold = threshold
        self.bandwidth_abundant = bandwidth_abundant

    def route(
        self,
        total_length: int,
        cached_prefix_membrane: int = 0,
        cached_prefix_pd: int = 0,
    ) -> RoutingDecision:
        """Route a single request.

        Args:
            total_length: Total input length of the request (l_total).
            cached_prefix_membrane: Cached prefix length available in the Membrane
                cluster (l_membrane).
            cached_prefix_pd: Cached prefix length available in the PD cluster
                (l_pd).

        Returns:
            RoutingDecision with target cluster and incremental length.
        """
        if self.bandwidth_abundant:
            # When bandwidth is abundant, compute is the scarce resource.
            # Use the best cache across all clusters.
            best_prefix = max(cached_prefix_membrane, cached_prefix_pd)
            incremental = total_length - best_prefix
            target = "pd-p" if incremental <= self.threshold else "membrane"
            cross_cluster = (
                target == "pd-p" and best_prefix == cached_prefix_membrane
            ) or (target == "membrane" and best_prefix == cached_prefix_pd)
            return RoutingDecision(
                target=target,
                incremental_length=max(0, incremental),
                cached_prefix_length=best_prefix,
                cross_cluster_cache_transfer=cross_cluster,
            )
        else:
            # When bandwidth is scarce, evaluate each cluster independently.
            # Route to PD-P if its incremental length is within threshold,
            # otherwise offload to Membrane.
            incremental_pd = total_length - cached_prefix_pd
            if incremental_pd <= self.threshold:
                return RoutingDecision(
                    target="pd-p",
                    incremental_length=max(0, incremental_pd),
                    cached_prefix_length=cached_prefix_pd,
                    cross_cluster_cache_transfer=False,
                )
            incremental_membrane = total_length - cached_prefix_membrane
            return RoutingDecision(
                target="membrane",
                incremental_length=max(0, incremental_membrane),
                cached_prefix_length=cached_prefix_membrane,
                cross_cluster_cache_transfer=False,
            )
