"""Offload: route prefill based on length, bandwidth, and cost.

The :class:`Offload` engine decides whether a prefill should run
locally or be offloaded to a remote Membrane node. The decision
balances four signals:

* **Prompt length** — short prompts are cheap enough to compute
  locally; long prompts benefit from a beefier remote node.
* **Local GPU/memory load** — a heavily-loaded local node should
  offload to keep tail latency in check.
* **Candidate availability** — when no candidate nodes are
  supplied the engine falls back to local compute.
* **Memory headroom on the target** — the candidate score
  penalizes targets with little free memory.

The engine produces an :class:`OffloadResult` describing the
chosen target node, whether local compute was selected, the
estimated cost, and a human-readable reason — useful for logging
and post-hoc tuning.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.cost import CostModel
from membrane.node import Node


@dataclass(frozen=True)
class OffloadResult:
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
class OffloadConfig:
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


class Offload:
    """Decides whether to compute locally or offload prefill to a remote node.

    Decision factors:
        - Prompt length (short → local, long → remote)
        - Local GPU load (high → remote)
        - Bandwidth cost (expensive → local)
        - KV size estimate (large → remote with big memory)
    """

    def __init__(
        self,
        config: OffloadConfig | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            config: Threshold configuration. Defaults to
                ``short_prompt_threshold=512`` and
                ``local_load_threshold=0.8``.
            cost_model: Cost model for compute vs transfer
                comparison. A default :class:`CostModel` is used
                when ``None``.
        """
        self.config = config or OffloadConfig()
        self.cost_model = cost_model or CostModel()

    def decide(
        self,
        prompt_tokens: list[int],
        local_node: Node,
        candidate_nodes: list[Node],
    ) -> OffloadResult:
        """Select the best node for prefill computation."""
        length = len(prompt_tokens)
        cfg = self.config
        local_load = local_node.heartbeat()
        local_cost = self.cost_model.prefill_cost(length)

        local_result = self._local_choice(
            length=length,
            local_node=local_node,
            local_load=local_load,
            local_cost=local_cost,
            short_threshold=cfg.short_prompt_threshold,
            load_threshold=cfg.local_load_threshold,
            candidate_nodes=candidate_nodes,
        )
        if local_result is not None:
            return local_result

        best = min(candidate_nodes, key=self._score(length))
        return OffloadResult(
            target_node_id=best.node_id,
            local_compute=False,
            estimated_cost_seconds=self.cost_model.prefill_cost(length),
            reason="offloaded to lower-load node",
        )

    def _local_choice(
        self,
        length: int,
        local_node: Node,
        local_load: float,
        local_cost: float,
        short_threshold: int,
        load_threshold: float,
        candidate_nodes: list[Node],
    ) -> OffloadResult | None:
        """Return a local-decision result, or ``None`` to fall through.

        Args:
            length: Prompt length in tokens.
            local_node: Local :class:`Node`.
            local_load: Local node load ratio in ``[0, 1]``.
            local_cost: Estimated prefill cost on the local node.
            short_threshold: ``config.short_prompt_threshold``.
            load_threshold: ``config.local_load_threshold``.
            candidate_nodes: Remote candidates (used to decide
                whether the fall-through applies).

        Returns:
            OffloadResult | None: Local compute result when the
            engine prefers to stay local, or ``None`` when
            offload should be considered.
        """
        if length <= short_threshold and local_load < load_threshold:
            return OffloadResult(
                target_node_id=local_node.node_id,
                local_compute=True,
                estimated_cost_seconds=local_cost,
                reason="short prompt and low local load",
            )
        if not candidate_nodes:
            return OffloadResult(
                target_node_id=local_node.node_id,
                local_compute=True,
                estimated_cost_seconds=local_cost,
                reason="no candidate nodes available",
            )
        return None

    def _score(self, length: int):
        """Return a scoring function for selecting among candidates.

        Lower score is better. Combines remote compute cost and
        memory headroom penalty.
        """

        def _score_node(node: Node) -> float:
            load = node.heartbeat()
            memory_headroom = 1.0 - load
            remote_cost = self.cost_model.prefill_cost(length)
            return remote_cost * load + (1.0 / (memory_headroom + 0.01))

        return _score_node


__all__ = ["Offload", "OffloadConfig", "OffloadResult"]
