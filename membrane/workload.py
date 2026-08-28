"""WorkloadAnalyzer: detect patterns in access logs.

This module defines :class:`WorkloadAnalyzer`, a small helper
that summarizes a stream of ``content_hash`` accesses. It exposes
three operations:

* :meth:`analyze_patterns` — content hash → normalized
  frequency.
* :meth:`top_patterns` — the ``k`` most frequent hashes.
* :meth:`reuse_ratio` — fraction of accesses that hit an
  already-seen hash.

The analyzer is stateless and intentionally cheap; it is
intended for ad-hoc inspection of access logs and for
bootstrapping the prediction / promotion layers with empirical
frequencies.
"""

import logging

logger = logging.getLogger(__name__)


from collections import Counter


class WorkloadAnalyzer:
    """Analyzes access logs to detect repeated prefix patterns."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass

    def analyze_patterns(self, access_log: list[str]) -> dict[str, float]:
        """Compute a content_hash → normalized frequency map.

        Args:
            access_log: Ordered list of accessed content
                hashes.

        Returns:
            dict[str, float]: Map of ``content_hash`` to its
            normalized frequency in ``[0.0, 1.0]``. Empty when
            ``access_log`` is empty.
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
        """Return the top-``k`` most frequent patterns.

        Args:
            access_log: Ordered list of accessed content
                hashes.
            k: Number of top patterns to return.

        Returns:
            list[tuple[str, float]]: ``(content_hash,
            frequency)`` tuples sorted in descending order of
            frequency. May be shorter than ``k`` when the log
            contains fewer distinct hashes.
        """
        frequencies = self.analyze_patterns(access_log)
        sorted_items = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
        return sorted_items[:k]

    def reuse_ratio(self, access_log: list[str]) -> float:
        """Compute the fraction of accesses that are repeats.

        Args:
            access_log: Ordered list of accessed content
                hashes.

        Returns:
            float: Ratio in ``[0.0, 1.0]``. ``1.0`` means
            every access is a repeat of an earlier one;
            ``0.0`` means every access is unique.
        """
        if not access_log:
            return 0.0
        unique = len(set(access_log))
        total = len(access_log)
        return (total - unique) / total if total > unique else 0.0
