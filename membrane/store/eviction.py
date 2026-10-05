"""Eviction policies: which resident fragments leave first when a node is full.

A policy only *orders* candidates; :class:`~membrane.node.Node` removes
them until enough bytes are free. Expired fragments always go first and
graph neighbours of evicted fragments last, whatever the policy.

* :class:`WeightedLRU` (default): least recently used, weighted by
  ``reuse_score``.
* :class:`FrequencyLRU`: least frequently used per a
  :class:`~membrane.decision.TinyLFU` sketch, ties broken by recency.
"""

from typing import Protocol, override

from membrane.constants import EVICTION_REUSE_EPSILON
from membrane.decision import TinyLFU
from membrane.fragment import Fragment


class EvictionPolicy(Protocol):
    """Orders eviction candidates, first victim first."""

    def order(self, candidates: list[tuple[str, Fragment]], access_times: dict[str, float], now: float) -> list[str]:
        """Return candidate hashes in eviction order.

        Args:
            candidates: ``(content_hash, fragment)`` pairs.
            access_times: Last access time per hash.
            now: Current time (the default for never-accessed hashes).

        Returns:
            list[str]: Hashes, first to evict first.
        """
        ...

    def touch(self, content_hash: str) -> None:
        """Record a hit, so the policy can favour keeping the hash.

        Args:
            content_hash: The accessed hash.
        """
        ...


class WeightedLRU:
    """Evict by ``last_access / (reuse_score + ε)``, lowest first."""

    def order(self, candidates: list[tuple[str, Fragment]], access_times: dict[str, float], now: float) -> list[str]:
        """Return candidates by ascending weighted recency.

        Args:
            candidates: ``(content_hash, fragment)`` pairs.
            access_times: Last access time per hash.
            now: Default access time.

        Returns:
            list[str]: Hashes, first to evict first.
        """

        def score(item: tuple[str, Fragment]) -> float:
            content_hash, fragment = item
            # Older access and lower reuse both lower the score.
            return access_times.get(content_hash, now) / (fragment.reuse_score + EVICTION_REUSE_EPSILON)

        return [h for h, _ in sorted(candidates, key=score)]

    def touch(self, content_hash: str) -> None:
        """Recency is tracked by the table; nothing to record.

        Args:
            content_hash: The accessed hash.
        """


class FrequencyLRU(WeightedLRU):
    """Evict the least frequently used first (TinyLFU estimate), then by recency.

    Attributes:
        sketch: Frequency sketch updated on every hit.
    """

    def __init__(self, sketch: TinyLFU | None = None) -> None:
        """Wrap a frequency sketch.

        Args:
            sketch: The sketch; a default :class:`TinyLFU` when ``None``.
        """
        self.sketch = sketch or TinyLFU()

    @override
    def order(self, candidates: list[tuple[str, Fragment]], access_times: dict[str, float], now: float) -> list[str]:
        """Return candidates by ascending (frequency, recency).

        Args:
            candidates: ``(content_hash, fragment)`` pairs.
            access_times: Last access time per hash.
            now: Default access time.

        Returns:
            list[str]: Hashes, first to evict first.
        """
        return [
            h
            for h, _ in sorted(
                candidates, key=lambda item: (self.sketch.estimate(item[0]), access_times.get(item[0], now))
            )
        ]

    @override
    def touch(self, content_hash: str) -> None:
        """Count a hit in the sketch.

        Args:
            content_hash: The accessed hash.
        """
        self.sketch.touch(content_hash)


__all__ = ["EVICTION_REUSE_EPSILON", "EvictionPolicy", "FrequencyLRU", "WeightedLRU"]
