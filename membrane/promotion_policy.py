"""PromotionPolicy: promote fragments to replicas based on demand."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of a promotion evaluation.

    Attributes:
        should_promote: Whether the fragment should be replicated.
        target_replicas: List of replica node IDs to receive copies.
        reason: Human-readable promotion reason.
    """

    should_promote: bool
    target_replicas: list[str]
    reason: str


@dataclass(frozen=True)
class PromotionConfig:
    """Configuration for promotion policy thresholds.

    Attributes:
        reuse_threshold: Minimum reuse_score for promotion.
        demand_threshold: Minimum access count from distinct regions.
        max_replicas: Maximum number of replicas allowed.
    """

    reuse_threshold: float = 0.7
    demand_threshold: int = 3
    max_replicas: int = 3


class PromotionPolicy:
    """Decides when and where to promote fragments to replicas.

    Rules:
    - reuse_score exceeds threshold
    - Multi-region access count exceeds threshold
    - Maximum 2–3 replicas enforced.
    """

    def __init__(self, config: PromotionConfig | None = None) -> None:
        """Initialize the promotion policy.

        Args:
            config: Promotion thresholds. Defaults to reuse_threshold=0.7.
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
            access_counts_by_region: Map of region -> access count.
            existing_replicas: Current replica node IDs.

        Returns:
            PromotionDecision with targets and reasoning.
        """
        cfg = self.config
        if fragment.reuse_score < cfg.reuse_threshold:
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="reuse_score below threshold",
            )

        total_demand = sum(access_counts_by_region.values())
        if total_demand < cfg.demand_threshold:
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="demand below threshold",
            )

        if len(existing_replicas) >= cfg.max_replicas:
            return PromotionDecision(
                should_promote=False,
                target_replicas=[],
                reason="max replicas reached",
            )

        # Select top-demand regions not already replicated
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
