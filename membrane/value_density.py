"""ValueDensity: compute importance × expected reuse score."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment


class ValueDensity:
    """Computes the economic value density of a fragment."""

    def compute(
        self,
        fragment: Fragment,
        access_history: list[str],
        importance: float = 1.0,
    ) -> float:
        """Compute value density for a fragment.

        Formula: importance × expected_reuse

        Args:
            fragment: Fragment to evaluate.
            access_history: Ordered list of recent accesses.
            importance: Importance multiplier (default 1.0).

        Returns:
            Value density score.
        """
        if not access_history:
            expected_reuse = fragment.reuse_score
        else:
            count = access_history.count(fragment.content_hash)
            recency_bonus = 0.1 if fragment.content_hash == access_history[-1] else 0.0
            expected_reuse = min(
                1.0, fragment.reuse_score + count * 0.05 + recency_bonus
            )
        return importance * expected_reuse
