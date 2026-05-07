"""Predictor: lightweight heuristic model for proactive memory staging."""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.model.profiler import kv_size_mib


class Predictor:
    """Lightweight heuristic predictor for KV size, reuse probability, and optimal region."""

    def __init__(self, kv_size_bias: float = 1.0) -> None:
        """Initialize the predictor.

        Args:
            kv_size_bias: Multiplicative bias for KV size estimation.
        """
        """Initialize the predictor.

        Args:
            kv_size_bias: Multiplicative bias for KV size estimation.
        """
        self.kv_size_bias = kv_size_bias

    def predict_kv_size(self, prompt_tokens: list[int]) -> float:
        """Predict KV cache size for a prompt.

        Args:
            prompt_tokens: Input token IDs.

        Returns:
            Estimated KV size in MiB.
        """
        return kv_size_mib(len(prompt_tokens)) * self.kv_size_bias

    def predict_reuse_probability(
        self,
        content_hash: str,
        session_history: list[str],
    ) -> float:
        """Predict likelihood of reuse based on session history.

        Args:
            content_hash: Fragment hash to evaluate.
            session_history: Recently accessed hashes in this session.

        Returns:
            Estimated reuse probability in [0.0, 1.0].
        """
        if not session_history:
            return 0.0
        recent = session_history[-10:]
        count = recent.count(content_hash)
        return min(1.0, count / len(recent))

    def predict_optimal_region(
        self,
        prompt_tokens: list[int],
        nodes: list[MembraneNode],
    ) -> str:
        """Predict the optimal node for a prompt based on load and capacity.

        Args:
            prompt_tokens: Input token IDs.
            nodes: Candidate nodes.

        Returns:
            Node identifier of the predicted optimal node.
        """
        if not nodes:
            return ""
        # Choose node with lowest heartbeat (memory pressure)
        best = min(nodes, key=lambda node: node.heartbeat())
        return best.node_id
