"""DynamicRoleManager: node dynamically switches role based on system state.

This module implements a small role-assignment policy that
:class:`~membrane.membrane_node.MembraneNode` instances can use to
decide whether to behave as a *memory host*, *prefill worker*, or
*decode worker* in a disaggregated inference cluster.

The manager is intentionally simple: it inspects the node's
current memory pressure (via
:meth:`MembraneNode.heartbeat`) and the cluster's average GPU
load, then picks a role from a fixed decision table. Callers that
need richer policies can subclass and override
:meth:`DynamicRoleManager.evaluate_role`.

Heuristic summary:

    * Memory pressure ``< 0.3`` and GPU load ``> 0.7`` →
      ``PREFILL_WORKER`` (lots of compute, little memory needed).
    * Memory pressure ``> 0.7`` and GPU load ``< 0.3`` →
      ``MEMORY_HOST`` (lots of memory, little compute available).
    * Otherwise: pick the role that matches the current
      cluster-wide deficit (compute-dominant → ``DECODE_WORKER``,
      memory-dominant → ``MEMORY_HOST``).
"""

import logging

logger = logging.getLogger(__name__)


from enum import Enum

from membrane.membrane_node import MembraneNode


class NodeRole(Enum):
    """Possible roles for a :class:`MembraneNode`.

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


class DynamicRoleManager:
    """Evaluates and assigns dynamic roles to nodes.

    The manager is stateless; instances are safe to share across
    threads as long as the underlying ``MembraneNode`` references
    are themselves safe to query concurrently.
    """

    def __init__(self) -> None:
        """Initialize the role manager."""
        pass

    def evaluate_role(
        self,
        node: MembraneNode,
        system_state: SystemState,
    ) -> NodeRole:
        """Decide the best role for ``node`` given ``system_state``.

        Heuristic:

        * High memory pressure + low GPU load → ``MEMORY_HOST``.
        * Low memory pressure + high GPU load → ``PREFILL_WORKER``.
        * Otherwise: role matching the cluster-wide deficit
          (compute-dominant → ``DECODE_WORKER``, otherwise
          ``MEMORY_HOST``).

        Args:
            node: Node to evaluate.
            system_state: Current cluster state.

        Returns:
            NodeRole: Recommended role.
        """
        # heart beat reports memory pressure in [0, 1].
        memory_pressure = node.heartbeat()
        gpu_load = system_state.average_gpu_load

        # Lots of GPU available, plenty of memory room →
        # prefill (compute-heavy) worker.
        if memory_pressure < 0.3 and gpu_load > 0.7:
            return NodeRole.PREFILL_WORKER

        # Memory-constrained node with idle GPU → host fragments.
        if memory_pressure > 0.7 and gpu_load < 0.3:
            return NodeRole.MEMORY_HOST

        # Otherwise pick based on the dominant cluster-wide
        # deficit.
        if system_state.total_compute_demand > system_state.total_memory_demand:
            return NodeRole.DECODE_WORKER

        return NodeRole.MEMORY_HOST
