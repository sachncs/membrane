"""Roles: node dynamically switches role based on system state.

Heuristic:

    * Memory pressure ``< 0.3`` and GPU load ``> 0.7`` →
      ``PREFILL_WORKER`` (lots of compute, little memory needed).
    * Memory pressure ``> 0.7`` and GPU load ``< 0.3`` →
      ``MEMORY_HOST`` (lots of memory, little compute available).
    * Otherwise: pick the role that matches the current
      cluster-wide deficit (compute-dominant → ``DECODE_WORKER``,
      memory-dominant → ``MEMORY_HOST``).

The decision table is published at module scope as
:meth:`Roles.ROLE_TABLE` and is consulted from
:meth:`Roles.evaluate_role` via the helper
:meth:`Roles._classify_state` that returns the row label
for a given (memory_pressure, gpu_load) pair. The
default-vs-deficit fallback chain is preserved.
"""

import logging

logger = logging.getLogger(__name__)


from enum import Enum
from typing import ClassVar

from membrane.node import Node


class NodeRole(Enum):
    """Possible roles for a :class:`Node`.

    * ``MEMORY_HOST`` — primarily stores fragments and serves
      reads.
    * ``PREFILL_WORKER`` — runs prefill computation; may evict
      aggressively to free compute resources.
    * ``DECODE_WORKER`` — runs decode computation; serves
      cached fragments at low latency.
    """

    MEMORY_HOST = "memory_host"
    PREFILL_WORKER = "prefill_worker"
    DECODE_WORKER = "decode_worker"


class SystemState:
    """Snapshot of system-wide state for role decisions.

    Attributes:
        total_compute_demand: Normalized compute demand across
            the cluster, in ``[0, 1]``.
        total_memory_demand: Normalized memory demand across the
            cluster, in ``[0, 1]``.
        average_gpu_load: Average GPU utilization, in ``[0, 1]``.
    """

    def __init__(
        self,
        total_compute_demand: float = 0.5,
        total_memory_demand: float = 0.5,
        average_gpu_load: float = 0.5,
    ) -> None:
        """Initialize the system state snapshot.

        Args:
            total_compute_demand: Cluster-wide compute demand.
                Defaults to ``0.5`` (balanced).
            total_memory_demand: Cluster-wide memory demand.
                Defaults to ``0.5``.
            average_gpu_load: Average GPU utilization. Defaults
                to ``0.5``.
        """
        self.total_compute_demand = total_compute_demand
        self.total_memory_demand = total_memory_demand
        self.average_gpu_load = average_gpu_load


class Roles:
    """Evaluates and assigns dynamic roles to nodes.

    The decision table maps a (memory_pressure, gpu_load)
    classification to one of three decisions:

    * ``"prefill"`` — call it for the PREFILL_WORKER role.
    * ``"memory"`` — call it for the MEMORY_HOST role.
    * ``"deficit"`` — defer to the cluster-wide deficit
      column (``compute_demand`` vs ``memory_demand``).
    """

    #: Classification table consulted before falling through to the
    #: deficit-based decision. The key ``"default"`` matches every
    #: (memory_pressure, gpu_load) pair that does not appear earlier.
    ROLE_TABLE: ClassVar[dict[str, NodeRole]] = {
        "default": NodeRole.MEMORY_HOST,
        "prefill": NodeRole.PREFILL_WORKER,
        "memory": NodeRole.MEMORY_HOST,
        "deficit": NodeRole.DECODE_WORKER,
    }

    def __init__(self) -> None:
        """Initialize the role manager."""
        pass

    @staticmethod
    def classify_state(memory_pressure: float, gpu_load: float) -> str:
        """Map (memory_pressure, gpu_load) to a role-table row label.

        Args:
            memory_pressure: Memory pressure ratio in ``[0, 1]``.
            gpu_load: GPU load ratio in ``[0, 1]``.

        Returns:
            str: One of ``"prefill"`` / ``"memory"`` / ``"default"``.
        """
        if memory_pressure < 0.3 and gpu_load > 0.7:
            return "prefill"
        if memory_pressure > 0.7 and gpu_load < 0.3:
            return "memory"
        return "default"

    def evaluate_role(
        self,
        node: Node,
        system_state: SystemState,
    ) -> NodeRole:
        """Decide the best role for ``node`` given ``system_state``.

        Args:
            node: Node to evaluate.
            system_state: Current cluster state.

        Returns:
            NodeRole: Recommended role.
        """
        memory_pressure = node.heartbeat()
        gpu_load = system_state.average_gpu_load

        row = self.classify_state(memory_pressure, gpu_load)
        if row != "default":
            return self.ROLE_TABLE[row]

        if system_state.total_compute_demand > system_state.total_memory_demand:
            return self.ROLE_TABLE["deficit"]
        return self.ROLE_TABLE["default"]


__all__ = ["NodeRole", "Roles", "SystemState"]
