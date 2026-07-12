"""OffloadDecisionEngine: route prefill based on length, bandwidth, and cost.

This module defines :class:`OffloadDecisionEngine`, which decides
whether a prefill should run locally or be offloaded to a remote
Membrane node. The decision balances four signals:

* **Prompt length** — short prompts are cheap enough to compute
  locally; long prompts benefit from a beefier remote node.
* **Local GPU/memory load** — a heavily-loaded local node should
  offload to keep tail latency in check.
* **Candidate availability** — when no candidate nodes are
  supplied the engine falls back to local compute.
* **Memory headroom on the target** — the candidate score
  penalizes targets with little free memory.

The engine produces an :class:`OffloadDecision` describing the
chosen target node, whether local compute was selected, the
estimated cost, and a human-readable reason — useful for logging
and post-hoc tuning.
"""

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
        local_compute: True if local node should compute, False
            if the prefill should be offloaded.
        estimated_cost_seconds: Estimated end-to-end latency in
            seconds.
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
        short_prompt_threshold: Maximum number of tokens
            considered "short" for local compute. Prompts above
            this length are eligible for offload.
        local_load_threshold: Maximum local node load (heart
            beat in ``[0, 1]``) below which the engine prefers
            local compute.
    """

    short_prompt_threshold: int = 512
    local_load_threshold: float = 0.8


class OffloadDecisionEngine:
    """Decides whether to compute locally or offload prefill to a remote node.

    Decision factors:
        - Prompt length (short → local, long → remote)
        - Local GPU load (high → remote)
        - Bandwidth cost (expensive → local)
        - KV size estimate (large → remote with big memory)
    """

    def __init__(
        self,
        config: OffloadDecisionConfig | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            config: Threshold configuration. Defaults to
                ``short_prompt_threshold=512`` and
                ``local_load_threshold=0.8``.
            cost_model: Cost model for compute vs transfer
                comparison. A default
                :class:`~membrane.cost_model.CostModel` is used
                when ``None``.
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

        Uses the cost model to compare local compute cost
        against remote compute cost adjusted for target node
        load and memory headroom.

        Args:
            prompt_tokens: Input token IDs.
            local_node: Node receiving the request.
            candidate_nodes: Other nodes that could perform
                prefill.

        Returns:
            OffloadDecision: Selected target, whether the
            decision is local, the estimated cost, and the
            reason.
        """
        length = len(prompt_tokens)
        local_load = local_node.heartbeat()
        local_cost = self.cost_model.precompute_cost_seconds(length)
        cfg = self.config

        # Fast path: short prompts on a quiet local node are
        # handled locally.
        if length <= cfg.short_prompt_threshold and local_load < cfg.local_load_threshold:
            return OffloadDecision(
                target_node_id=local_node.node_id,
                local_compute=True,
                estimated_cost_seconds=local_cost,
                reason="short prompt and low local load",
            )

        if not candidate_nodes:
            # No remote candidates; stay local even though it
            # may not be the cheapest choice.
            return OffloadDecision(
                target_node_id=local_node.node_id,
                local_compute=True,
                estimated_cost_seconds=local_cost,
                reason="no candidate nodes available",
            )

        # Score each candidate by combined compute cost and
        # memory pressure. The 1/(headroom + 0.01) term grows
        # rapidly as headroom shrinks, pushing selection toward
        # nodes with more spare memory.
        def score(node: MembraneNode) -> float:
            """Score offload target (lower is better)."""
            load = node.heartbeat()
            memory_headroom = 1.0 - load
            remote_cost = self.cost_model.precompute_cost_seconds(length)
            # Penalize high load and low memory headroom.
            return remote_cost * load + (1.0 / (memory_headroom + 0.01))

        best = min(candidate_nodes, key=score)
        best_cost = self.cost_model.precompute_cost_seconds(length)

        return OffloadDecision(
            target_node_id=best.node_id,
            local_compute=False,
            estimated_cost_seconds=best_cost,
            reason="offloaded to lower-load node",
        )
