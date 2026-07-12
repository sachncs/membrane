"""Predictor: lightweight heuristic model for proactive memory staging.

This module defines :class:`Predictor`, a small utility that
provides three quick estimates useful for proactive memory
staging decisions:

* :meth:`predict_kv_size` — expected KV cache footprint in MiB
  for a prompt of a given length, derived from the analytical
  :func:`membrane.model.profiler.kv_size_mib` formula.
* :meth:`predict_reuse_probability` — heuristic probability that
  a fragment will be reused based on recent session activity.
* :meth:`predict_optimal_region` — best node to serve a prompt
  based on current memory pressure.

The predictor is intentionally lightweight: it contains no
learned weights and is safe to call from request hot paths.
For richer predictions, layer a learned model that consumes the
same inputs and produces scores in ``[0, 1]``.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.model.profiler import kv_size_mib


class Predictor:
    """Lightweight heuristic predictor for KV size, reuse probability, and optimal region.

    Attributes:
        kv_size_bias: Multiplicative bias applied to KV size
            estimates, useful for inflating or deflating the
            expected footprint to match a specific deployment.
    """

    def __init__(self, kv_size_bias: float = 1.0) -> None:
        """Initialize the predictor.

        Args:
            kv_size_bias: Multiplicative bias for KV size
                estimation. Defaults to ``1.0`` (no bias).
        """
        self.kv_size_bias = kv_size_bias

    def predict_kv_size(self, prompt_tokens: list[int]) -> float:
        """Predict KV cache size for a prompt.

        Args:
            prompt_tokens: Input token IDs.

        Returns:
            float: Estimated KV size in MiB after applying
            ``kv_size_bias``.
        """
        return kv_size_mib(len(prompt_tokens)) * self.kv_size_bias

    def predict_reuse_probability(
        self,
        content_hash: str,
        session_history: list[str],
    ) -> float:
        """Predict likelihood of reuse based on session history.

        Uses only the last 10 accesses of the session — short
        recency windows react quickly to changing workloads at
        the cost of slightly higher variance.

        Args:
            content_hash: Fragment hash to evaluate.
            session_history: Recently accessed hashes in this
                session. May be empty.

        Returns:
            float: Estimated reuse probability in ``[0.0, 1.0]``.
            Returns ``0.0`` when ``session_history`` is empty.
        """
        if not session_history:
            return 0.0
        recent = session_history[-10:]
        count = recent.count(content_hash)
        # Normalize by the window length and clamp to [0, 1] so
        # the result is always a valid probability.
        return min(1.0, count / len(recent))

    def predict_optimal_region(
        self,
        prompt_tokens: list[int],
        nodes: list[MembraneNode],
    ) -> str:
        """Predict the optimal node for a prompt based on load and capacity.

        The current heuristic picks the node with the lowest
        memory pressure (``heartbeat()``). It does not consider
        GPU load or latency; callers that need a richer signal
        should use :class:`~membrane.economic_router.EconomicRouter`
        or :class:`~membrane.joint_optimizer.JointOptimizer`
        instead.

        Args:
            prompt_tokens: Input token IDs. Currently unused by
                the heuristic but accepted for forward
                compatibility.
            nodes: Candidate nodes.

        Returns:
            str: Node identifier of the predicted optimal node,
            or an empty string when ``nodes`` is empty.
        """
        if not nodes:
            return ""
        # Choose node with lowest heartbeat (memory pressure).
        best = min(nodes, key=lambda node: node.heartbeat())
        return best.node_id
