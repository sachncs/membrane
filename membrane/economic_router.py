"""EconomicRouter: route to argmax(value_density − cost) node.

All cost components are normalized to [0, 1] using configurable maximum
reference values, then combined with configurable weights.  This ensures
that no single dimension dominates the score because of scale differences.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.node_telemetry import NodeTelemetry
from membrane.value_density import ValueDensity


@dataclass(frozen=True)
class EconomicRouterConfig:
    """Configuration for cost-component normalization and weighting.

    Every cost component is first clamped to [0, 1] by dividing by its
    ``max_*`` reference value, then multiplied by its ``weight_*``.  The
    final cost is the weighted sum of the normalized components.

    Attributes:
        max_latency_ms: Latency value that maps to cost = 1.0.
        weight_latency: Weight of the normalized latency term.
        weight_bandwidth: Weight of the normalized bandwidth cost term.
        weight_gpu: Weight of the normalized GPU load term.
        weight_memory: Weight of the normalized memory pressure term.
    """

    max_latency_ms: float = 5000.0
    weight_latency: float = 1.0
    weight_bandwidth: float = 1.0
    weight_gpu: float = 1.0
    weight_memory: float = 1.0


class EconomicRouter:
    """Routes fragments to the node maximizing value density minus cost.

    Cost is a weighted sum of normalized telemetry dimensions so that no
    single dimension dominates because of inconsistent units.
    """

    def __init__(
        self,
        value_density: ValueDensity | None = None,
        config: EconomicRouterConfig | None = None,
    ) -> None:
        """Initialize with optional value density calculator and config.

        Args:
            value_density: ValueDensity instance.
            config: Normalization and weighting parameters.
        """
        self.value_density = value_density or ValueDensity()
        self.config = config or EconomicRouterConfig()

    def route(
        self,
        fragment: Fragment,
        candidate_node_ids: list[str],
        telemetry_map: dict[str, NodeTelemetry],
        access_history: list[str],
    ) -> str:
        """Select the best node for a fragment.

        Args:
            fragment: Fragment to place.
            candidate_node_ids: Candidate node identifiers.
            telemetry_map: Node ID -> telemetry snapshot.
            access_history: Recent access history for reuse estimation.

        Returns:
            Selected node identifier, or empty string if no candidate.
        """
        if not candidate_node_ids:
            return ""

        vd = self.value_density.compute(fragment, access_history)
        cfg = self.config

        def normalized_cost(node_id: str) -> float:
            """Compute weighted normalized cost for a node."""
            telem = telemetry_map.get(node_id)
            if telem is None:
                return float("inf")

            latency_norm = min(1.0, telem.latency_ms / cfg.max_latency_ms)
            bandwidth_norm = min(1.0, telem.bandwidth_cost)
            gpu_norm = min(1.0, telem.gpu_load)
            memory_norm = min(1.0, telem.memory_pressure)

            return (
                cfg.weight_latency * latency_norm
                + cfg.weight_bandwidth * bandwidth_norm
                + cfg.weight_gpu * gpu_norm
                + cfg.weight_memory * memory_norm
            )

        best = max(candidate_node_ids, key=lambda nid: vd - normalized_cost(nid))
        return best
