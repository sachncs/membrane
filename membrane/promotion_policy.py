"""PromotionPolicy: promote fragments to replicas based on demand.

This module defines :class:`PromotionPolicy` and its supporting
:class:`PromotionDecision` and :class:`PromotionConfig`
dataclasses. The policy decides whether a fragment should be
replicated to additional regions, and if so, to which ones.

Three conditions must all be satisfied for promotion:

1. The fragment's ``reuse_score`` is at or above
   ``config.reuse_threshold``.
2. Total cross-region demand is at or above
   ``config.demand_threshold``.
3. The fragment is not already replicated to the maximum
   number of regions.

When promotion is approved, the policy picks the highest-demand
regions that are *not* already replica targets, up to the
remaining replica slots.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of a promotion evaluation.

    Attributes:
        should_promote: Whether the fragment should be
            replicated.
        target_replicas: List of replica node IDs to receive
            copies. Empty when ``should_promote`` is False.
        reason: Human-readable promotion reason.
    """

    should_promote: bool
    target_replicas: list[str]
    reason: str


@dataclass(frozen=True)
class PromotionConfig:
    """Configuration for promotion policy thresholds.

    Attributes:
        reuse_threshold: Minimum ``reuse_score`` for promotion.
        demand_threshold: Minimum total access count across all
            regions for promotion.
        max_replicas: Maximum number of replicas allowed per
            fragment.
    """

    reuse_threshold: float = 0.7
    demand_threshold: int = 3
    max_replicas: int = 3


class PromotionPolicy:
    """Decides when and where to promote fragments to replicas.

    Rules:
        - ``reuse_score`` exceeds threshold
        - Multi-region access count exceeds threshold
        - Maximum 2-3 replicas enforced
    """

    def __init__(self, config: PromotionConfig | None = None) -> None:
        """Initialize the promotion policy.

        Args:
            config: Promotion thresholds. A default
                :class:`PromotionConfig` is used when ``None``.
        """
        self.config = config or PromotionConfig()

    def evaluate(
        self,
        fragment: Fragment,
        access_counts_by_region: dict[str, int],
        existing_replicas: list[str],
    ) -> PromotionDecision:
        """Evaluate whether a fragment should be promoted.

        Args:
            fragment: Candidate fragment.
            access_counts_by_region: Map of ``region -> access
                count``.
            existing_replicas: Current replica node IDs.

        Returns:
            PromotionDecision: Selected targets and a reason
            explaining the verdict (positive or negative).
        """
        cfg = self.config

        # Gate 1: hot enough to be worth replicating?
        if fragment.reuse_score < cfg.reuse_threshold:
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="reuse_score below threshold",
            )

        # Gate 2: enough cross-region demand to justify a copy?
        total_demand = sum(access_counts_by_region.values())
        if total_demand < cfg.demand_threshold:
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="demand below threshold",
            )

        # Gate 3: room to add replicas?
        if len(existing_replicas) >= cfg.max_replicas:
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="max replicas reached",
            )

        # Select top-demand regions that are not already
        # replicated, up to the number of available slots.
        sorted_regions = sorted(
            access_counts_by_region.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        slots = cfg.max_replicas - len(existing_replicas)
        targets: list[str] = []
        for region, count in sorted_regions:
            if region not in existing_replicas and count > 0:
                targets.append(region)
                if len(targets) >= slots:
                    break

        if not targets:
            # Demand exists but no region is a viable target
            # (e.g., all are already replicated or have zero
            # recent accesses).
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="no suitable regions",
            )

        return PromotionDecision(
            should_promote=True,
            target_replicas=targets,
            reason="high reuse and multi-region demand",
        )
