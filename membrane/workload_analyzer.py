"""WorkloadAnalyzer: detect patterns in access logs."""

import logging

logger = logging.getLogger(__name__)


from collections import Counter


class WorkloadAnalyzer:
    """Analyzes access logs to detect repeated prefix patterns."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        """Initialize the analyzer."""
        pass

    def analyze_patterns(self, access_log: list[str]) -> dict[str, float]:
        """Compute frequency map for content hashes in an access log.

        Args:
            access_log: Ordered list of accessed content hashes.

        Returns:
            Map of content_hash -> normalized frequency in [0.0, 1.0].
        """
        if not access_log:
            return {}
        counts = Counter(access_log)
        total = len(access_log)
        return {h: count / total for h, count in counts.items()}

    def top_patterns(
        self,
        access_log: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the top-k most frequent patterns.

        Args:
            access_log: Ordered list of accessed content hashes.
            k: Number of top patterns to return.

        Returns:
            List of (content_hash, frequency) tuples sorted descending.
        """
        frequencies = self.analyze_patterns(access_log)
        sorted_items = sorted(
            frequencies.items(), key=lambda item: item[1], reverse=True
        )
        return sorted_items[:k]

    def reuse_ratio(self, access_log: list[str]) -> float:
        """Compute the fraction of accesses that are repeats.

        Args:
            access_log: Ordered list of accessed content hashes.

        Returns:
            Ratio in [0.0, 1.0] where 1.0 means all accesses are repeats.
        """
        if not access_log:
            return 0.0
        unique = len(set(access_log))
        total = len(access_log)
        return (total - unique) / total if total > unique else 0.0
