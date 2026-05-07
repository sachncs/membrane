"""NodeSelector: multi-criteria node selection for routing and placement.

Selects optimal nodes from a candidate set based on configurable
latency, load, memory, and bandwidth scoring.  Supports health
threshold filtering and topology-aware preference.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.node_telemetry import NodeTelemetry


@dataclass(frozen=True)
class NodeSelectorConfig:
    """Configuration for node-selection scoring.

    Every dimension is clamped to [0, 1] using ``max_*`` reference values,
    then multiplied by its ``weight_*``.  Nodes with any dimension above
    ``health_threshold`` are excluded entirely.

    Attributes:
        max_latency_ms: Latency value that maps to score = 1.0.
        max_gpu_load: GPU load value that maps to score = 1.0.
        max_memory_pressure: Memory pressure value that maps to score = 1.0.
        max_bandwidth_cost: Bandwidth cost value that maps to score = 1.0.
        weight_latency: Weight of the normalized latency term.
        weight_gpu: Weight of the normalized GPU load term.
        weight_memory: Weight of the normalized memory pressure term.
        weight_bandwidth: Weight of the normalized bandwidth cost term.
        health_threshold: Maximum allowed raw value for any dimension.
            A node exceeding this on *any* dimension is considered unhealthy.
    """

    max_latency_ms: float = 5000.0
    max_gpu_load: float = 1.0
    max_memory_pressure: float = 1.0
    max_bandwidth_cost: float = 1.0
    weight_latency: float = 1.0
    weight_gpu: float = 1.0
    weight_memory: float = 1.0
    weight_bandwidth: float = 1.0
    health_threshold: float = 0.95


class NodeSelector:
    """Selects the best node(s) from a candidate pool.

    Lower composite scores are better (minimizes cost).  Unhealthy nodes
    are filtered out before ranking.
    """

    def __init__(self, config: NodeSelectorConfig | None = None) -> None:
        self.config = config or NodeSelectorConfig()

    def select(
        self,
        candidate_node_ids: list[str],
        telemetry_map: dict[str, NodeTelemetry],
    ) -> str:
        """Return the single best node identifier.

        Args:
            candidate_node_ids: Candidate node identifiers.
            telemetry_map: Node ID -> telemetry snapshot.

        Returns:
            Best node identifier, or empty string if no healthy candidate.
        """
        healthy = self.filter_healthy(candidate_node_ids, telemetry_map)
        if not healthy:
            return ""
        return min(healthy, key=lambda nid: self.score(nid, telemetry_map))

    def select_top_n(
        self,
        candidate_node_ids: list[str],
        telemetry_map: dict[str, NodeTelemetry],
        n: int = 3,
    ) -> list[str]:
        """Return the top *n* best node identifiers in ascending score order.

        Args:
            candidate_node_ids: Candidate node identifiers.
            telemetry_map: Node ID -> telemetry snapshot.
            n: Maximum number of nodes to return.

        Returns:
            List of best node identifiers (may be fewer than *n*).
        """
        healthy = self.filter_healthy(candidate_node_ids, telemetry_map)
        if not healthy:
            return []
        scored = sorted(healthy, key=lambda nid: self.score(nid, telemetry_map))
        return scored[:n]

    def filter_healthy(
        self,
        candidate_node_ids: list[str],
        telemetry_map: dict[str, NodeTelemetry],
    ) -> list[str]:
        """Remove nodes that exceed the health threshold on any dimension.

        Args:
            candidate_node_ids: Candidate node identifiers.
            telemetry_map: Node ID -> telemetry snapshot.

        Returns:
            List of healthy node identifiers.
        """
        cfg = self.config
        threshold = cfg.health_threshold
        healthy: list[str] = []
        for nid in candidate_node_ids:
            telem = telemetry_map.get(nid)
            if telem is None:
                continue
            if (
                telem.gpu_load > threshold
                or telem.memory_pressure > threshold
                or telem.bandwidth_cost > threshold
                or (telem.latency_ms / cfg.max_latency_ms) > threshold
            ):
                continue
            healthy.append(nid)
        return healthy

    def score(
        self,
        node_id: str,
        telemetry_map: dict[str, NodeTelemetry],
    ) -> float:
        """Compute composite cost score for a node (lower is better).

        Args:
            node_id: Node to score.
            telemetry_map: Node ID -> telemetry snapshot.

        Returns:
            Composite cost.  ``inf`` if telemetry is missing.
        """
        telem = telemetry_map.get(node_id)
        if telem is None:
            return float("inf")

        cfg = self.config
        latency_norm = min(1.0, telem.latency_ms / cfg.max_latency_ms)
        gpu_norm = min(1.0, telem.gpu_load / cfg.max_gpu_load)
        memory_norm = min(1.0, telem.memory_pressure / cfg.max_memory_pressure)
        bandwidth_norm = min(1.0, telem.bandwidth_cost / cfg.max_bandwidth_cost)

        return (
            cfg.weight_latency * latency_norm
            + cfg.weight_gpu * gpu_norm
            + cfg.weight_memory * memory_norm
            + cfg.weight_bandwidth * bandwidth_norm
        )
