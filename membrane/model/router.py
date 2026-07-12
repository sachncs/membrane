"""Request router implementing the routing policies in Sections 3.3 and 3.4.3.

The router decides whether a request is handled locally (PD-P)
or offloaded to a Membrane cluster based on its incremental
uncached length and current bandwidth conditions.

Policies:

* **Bandwidth-scarce** (default): each cluster's incremental
  length is evaluated independently. PD-P is preferred when its
  incremental length fits within ``threshold``; otherwise the
  request is offloaded to Membrane.
* **Bandwidth-abundant**: compute is the scarce resource, so the
  router uses the *best* cached prefix across both clusters. The
  resulting decision may require cross-cluster cache transfer.

References:
    * "Prefill-as-a-Service", arXiv:2604.15039v2, §3.3 and
      §3.4.3.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    """Outcome of a routing decision.

    Attributes:
        target: ``"membrane"`` or ``"pd-p"``.
        incremental_length: Tokens that still need to be
            prefilled after accounting for the cached prefix.
        cached_prefix_length: Length of the cached prefix used
            to compute ``incremental_length``.
        cross_cluster_cache_transfer: True when the best
            prefix cache lives in the *other* cluster from the
            chosen target, indicating that a transfer will
            occur before prefill.
    """

    target: str
    incremental_length: int
    cached_prefix_length: int
    cross_cluster_cache_transfer: bool = False


class Router:
    """Length-based threshold router with optional prefix-cache awareness.

    Implements the policies described in:

    * Section 3.3 — "When ``l > t``, the request is routed to a
      Membrane cluster".
    * Section 3.4.3 — short-term routing under
      bandwidth-scarce/abundant prefix-cache regimes.

    Attributes:
        threshold: Routing threshold ``t`` in tokens.
        bandwidth_abundant: Whether to enable the
          bandwidth-abundant regime.
    """

    def __init__(
        self,
        threshold: int,
        bandwidth_abundant: bool = False,
    ):
        """Initialize the router.

        Args:
            threshold: Routing threshold ``t`` in tokens.
            bandwidth_abundant: If ``True``, consider the
                global-best cache and allow cross-cluster cache
                transfer. If ``False`` (default), evaluate
                caches independently per cluster.
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
            total_length: Total input length of the request
                (``l_total``).
            cached_prefix_membrane: Cached prefix length
                available in the Membrane cluster
                (``l_membrane``).
            cached_prefix_pd: Cached prefix length available in
                the PD cluster (``l_pd``).

        Returns:
            RoutingDecision: Target cluster, incremental
            length, and (for the bandwidth-abundant regime)
            whether a cross-cluster transfer is required.
        """
        if self.bandwidth_abundant:
            # When bandwidth is abundant, compute is the scarce
            # resource. Pick the best cache across both clusters
            # and route so the incremental length fits within
            # the threshold.
            best_prefix = max(cached_prefix_membrane, cached_prefix_pd)
            incremental = total_length - best_prefix
            target = "pd-p" if incremental <= self.threshold else "membrane"
            # cross_cluster is True when the best cache lives in
            # the *other* cluster than the chosen target.
            cross_cluster = (
                target == "pd-p" and best_prefix == cached_prefix_membrane
            ) or (target == "membrane" and best_prefix == cached_prefix_pd)
            return RoutingDecision(
                target=target,
                incremental_length=max(0, incremental),
                cached_prefix_length=best_prefix,
                cross_cluster_cache_transfer=cross_cluster,
            )
        # Bandwidth-scarce regime: each cluster evaluated
        # independently. PD-P wins when its incremental length
        # fits within the threshold.
        incremental_pd = total_length - cached_prefix_pd
        if incremental_pd <= self.threshold:
            return RoutingDecision(
                target="pd-p",
                incremental_length=max(0, incremental_pd),
                cached_prefix_length=cached_prefix_pd,
                cross_cluster_cache_transfer=False,
            )
        # Otherwise offload to Membrane using its own cached
        # prefix (no cross-cluster transfer in this regime).
        incremental_membrane = total_length - cached_prefix_membrane
        return RoutingDecision(
            target="membrane",
            incremental_length=max(0, incremental_membrane),
            cached_prefix_length=cached_prefix_membrane,
            cross_cluster_cache_transfer=False,
        )
