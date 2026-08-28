"""density: compute importance x expected reuse score.

This module provides :func:`density`, a free function that scores how
*worth keeping* a particular fragment is. The score is the product of
the fragment's intrinsic importance and an expected-reuse signal derived
from the producer-supplied ``reuse_score`` plus a lightweight demand
signal from the access history.

For richer reuse modeling, see :mod:`membrane.predict`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment


def density(
    fragment: Fragment,
    access_history: list[str],
    importance: float = 1.0,
) -> float:
    """Compute the value density of ``fragment``.

    The score is ``importance * expected_reuse``. ``expected_reuse``
    combines the fragment's intrinsic ``reuse_score`` with two demand
    signals derived from ``access_history``:

    * **Frequency**: ``count * 0.05``, where ``count`` is the number
      of times ``fragment.content_hash`` appears in the access history.
    * **Recency**: a flat ``+0.1`` bonus if the most recent access was
      for this fragment.

    The combined signal is clamped to ``[0, 1]`` before being
    multiplied by ``importance``.

    Args:
        fragment: The fragment to evaluate. Only its ``content_hash``
            and ``reuse_score`` are read.
        access_history: Ordered list of recently-accessed
            ``content_hash`` values, most recent last. May be empty,
            in which case the fragment's intrinsic ``reuse_score`` is
            used directly.
        importance: Importance multiplier applied to the expected-reuse
            signal. Defaults to ``1.0`` (no weighting).

    Returns:
        float: The value density score. Higher is better; bounded above
        by ``importance`` (since ``expected_reuse <= 1``).
    """
    if not access_history:
        return importance * fragment.reuse_score
    count = access_history.count(fragment.content_hash)
    recency_bonus = 0.1 if fragment.content_hash == access_history[-1] else 0.0
    expected_reuse = min(1.0, fragment.reuse_score + count * 0.05 + recency_bonus)
    return importance * expected_reuse
