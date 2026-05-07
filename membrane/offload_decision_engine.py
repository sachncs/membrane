"""OffloadDecisionEngine: route prefill based on length, bandwidth, and cost."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.cost_model import CostModel
from membrane.membrane_node import MembraneNode


@dataclass(frozen=True)
class OffloadDecision:
    """Outcome of an offload routing decision.

    Attributes:
        target_node_id: Node ID chosen for prefill.
        local_compute: True if local node should compute.
        estimated_cost_seconds: Estimated end-to-end latency.
        reason: Human-readable routing reason.
    """

    target_node_id: str
    local_compute: bool
    estimated_cost_seconds: float
    reason: str


@dataclass(frozen=True)
class OffloadDecisionConfig:
    """Configuration for offload decision thresholds.

    Attributes:
        short_prompt_threshold: Max tokens considered "short" for local compute.
        local_load_threshold: Max node load (0.0-1.0) before offloading.
    """

    short_prompt_threshold: int = 512
    local_load_threshold: float = 0.8


class OffloadDecisionEngine:
    """Decides whether to compute locally or offload prefill to a remote node.

    Decision factors:
    - Prompt length (short -> local, long -> remote)
    - Local GPU load (high -> remote)
    - Bandwidth cost (expensive -> local)
    - KV size estimate (large -> remote with big memory)
    """

    def __init__(
        self,
        config: OffloadDecisionConfig | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            config: Threshold configuration. Defaults to short_prompt_threshold=512.
            cost_model: Cost model for compute vs transfer comparison.
        """
        self.config = config or OffloadDecisionConfig()
        self.cost_model = cost_model or CostModel()

    def decide(
        self,
        prompt_tokens: list[int],
        local_node: MembraneNode,
        candidate_nodes: list[MembraneNode],
    ) -> OffloadDecision:
        """Select the best node for prefill computation.

        Uses the cost model to compare local compute cost against remote
        compute cost adjusted for target node load and memory headroom.

        Args:
            prompt_tokens: Input token IDs.
            local_node: Node receiving the request.
            candidate_nodes: Other nodes that could perform prefill.

        Returns:
            OffloadDecision with target and cost estimate.
        """
        length = len(prompt_tokens)
        local_load = local_node.heartbeat()
        local_cost = self.cost_model.precompute_cost_seconds(length)
        cfg = self.config

        if length <= cfg.short_prompt_threshold and local_load < cfg.local_load_threshold:
            return OffloadDecision(
                target_node_id=local_node.node_id,
                local_compute=True,
                estimated_cost_seconds=local_cost,
                reason="short prompt and low local load",
            )

        if not candidate_nodes:
            return OffloadDecision(
                target_node_id=local_node.node_id,
                local_compute=True,
                estimated_cost_seconds=local_cost,
                reason="no candidate nodes available",
            )

        # Find candidate with lowest combined compute + memory pressure
        def score(node: MembraneNode) -> float:
            """Score offload target: lower is better."""
            load = node.heartbeat()
            memory_headroom = 1.0 - load
            remote_cost = self.cost_model.precompute_cost_seconds(length)
            # Penalize high load and low memory headroom
            return remote_cost * load + (1.0 / (memory_headroom + 0.01))

        best = min(candidate_nodes, key=score)
        best_cost = self.cost_model.precompute_cost_seconds(length)

        return OffloadDecision(
            target_node_id=best.node_id,
            local_compute=False,
            estimated_cost_seconds=best_cost,
            reason="offloaded to lower-load node",
        )
