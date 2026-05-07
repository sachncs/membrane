"""DynamicRoleManager: node dynamically switches role based on system state."""

import logging

logger = logging.getLogger(__name__)


from enum import Enum

from membrane.membrane_node import MembraneNode


class NodeRole(Enum):
    """Possible roles for a MembraneNode."""

    MEMORY_HOST = "memory_host"
    PREFILL_WORKER = "prefill_worker"
    DECODE_WORKER = "decode_worker"


class SystemState:
    """Snapshot of system-wide state for role decisions.

    Attributes:
        total_compute_demand: Normalized compute demand across the cluster.
        total_memory_demand: Normalized memory demand across the cluster.
        average_gpu_load: Average GPU utilization.
    """

    def __init__(
        self,
        total_compute_demand: float = 0.5,
        total_memory_demand: float = 0.5,
        average_gpu_load: float = 0.5,
    ) -> None:
        """Initialize system state.

        Args:
            total_compute_demand: Cluster-wide compute demand.
            total_memory_demand: Cluster-wide memory demand.
            average_gpu_load: Average GPU utilization.
        """
        self.total_compute_demand = total_compute_demand
        self.total_memory_demand = total_memory_demand
        self.average_gpu_load = average_gpu_load


class DynamicRoleManager:
    """Evaluates and assigns dynamic roles to nodes."""

    def __init__(self) -> None:
        """Initialize the role manager."""
        """Initialize the role manager."""
        pass

    def evaluate_role(
        self,
        node: MembraneNode,
        system_state: SystemState,
    ) -> NodeRole:
        """Decide the best role for a node given system state.

        Heuristic:
        - High memory, low GPU load -> MEMORY_HOST
        - Low memory, high GPU load -> PREFILL_WORKER or DECODE_WORKER
        - Balanced -> role matching system deficit

        Args:
            node: Node to evaluate.
            system_state: Current cluster state.

        Returns:
            Recommended NodeRole.
        """
        memory_pressure = node.heartbeat()
        gpu_load = system_state.average_gpu_load

        if memory_pressure < 0.3 and gpu_load > 0.7:
            return NodeRole.PREFILL_WORKER

        if memory_pressure > 0.7 and gpu_load < 0.3:
            return NodeRole.MEMORY_HOST

        if system_state.total_compute_demand > system_state.total_memory_demand:
            return NodeRole.DECODE_WORKER

        return NodeRole.MEMORY_HOST
