"""ValueDensity: compute importance × expected reuse score.

This module defines :class:`ValueDensity`, a small economic helper used
by the routing and placement layers to score how *worth keeping* a
particular fragment is.

The idea is intentionally simple: a fragment's value to the cluster
is the product of

* how important it is to its owner (``importance``), and
* how often it is expected to be reused in the near future
  (``expected_reuse``).

The result is a single scalar that downstream components (the
:class:`~membrane.promotion_policy.PromotionPolicy`,
:class:`~membrane.economic_router.EconomicRouter`, and the canonical
store) can use to break ties when capacity is constrained.

Assumptions:
    * ``access_history`` is provided by the caller (typically the
      :class:`~membrane.session_tracker.SessionTracker`) and is
      treated as authoritative; this class does not validate its
      ordering.
    * The reuse model is intentionally crude — a richer predictor
      lives in :mod:`membrane.predictor`. This module exists so that
      lightweight decisions can be made without paying for the
      heavier machinery.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment


class ValueDensity:
    """Computes the economic value density of a fragment.

    Stateless and safe to share across threads. Instances are cheap
    to construct; callers typically instantiate a single module-level
    instance.
    """

    def compute(
        self,
        fragment: Fragment,
        access_history: list[str],
        importance: float = 1.0,
    ) -> float:
        """Compute the value density of ``fragment``.

        The score is ``importance * expected_reuse``. ``expected_reuse``
        combines the fragment's intrinsic ``reuse_score`` with two
        demand signals derived from ``access_history``:

        * **Frequency**: ``count * 0.05``, where ``count`` is the
          number of times ``fragment.content_hash`` appears in the
          access history.
        * **Recency**: a flat ``+0.1`` bonus if the most recent access
          was for this fragment.

        The combined signal is clamped to ``[0, 1]`` before being
        multiplied by ``importance``.

        Args:
            fragment: The fragment to evaluate. Only its
                ``content_hash`` and ``reuse_score`` are read.
            access_history: Ordered list of recently-accessed
                ``content_hash`` values, most recent last. May be
                empty, in which case the fragment's intrinsic
                ``reuse_score`` is used directly.
            importance: Importance multiplier applied to the
                expected-reuse signal. Defaults to ``1.0`` (no
                weighting).

        Returns:
            float: The value density score. Higher is better; the
            value is bounded above by ``importance`` (since
            ``expected_reuse <= 1``).
        """
        if not access_history:
            # No demand signal available; trust the producer-supplied
            # reuse_score as the best prior on expected reuse.
            expected_reuse = fragment.reuse_score
        else:
            # Demand signal from observed accesses:
            # - count: how often the fragment has been touched
            # - recency_bonus: whether the most recent access was for it
            count = access_history.count(fragment.content_hash)
            recency_bonus = 0.1 if fragment.content_hash == access_history[-1] else 0.0
            expected_reuse = min(
                1.0, fragment.reuse_score + count * 0.05 + recency_bonus
            )
        return importance * expected_reuse
