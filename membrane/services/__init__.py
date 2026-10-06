"""Node services above storage: the memory API, routing, and background policies.

:class:`Services` builds them from :class:`ServiceOptions` and runs their
periodic tasks; :class:`~membrane.server.Server` owns one instance and the
HTTP routes reach it through :class:`~membrane.transport.context.AppContext`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from membrane.compat import ModelCompatibilityFingerprint, compat_hash
from membrane.runtime.lifecycle import PeriodicTask
from membrane.services.memory import MemoryService
from membrane.services.placement import ClusterView, PlacementService
from membrane.services.policies import OriginLink, Promoter, RolePolicy

logger = logging.getLogger(__name__)

#: Seconds between short-term routing adjustments.
ROUTE_ADJUST_INTERVAL_SEC = 5.0
#: Seconds between long-term routing re-fits.
ROUTE_REFIT_INTERVAL_SEC = 600.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceOptions:
    """Settings of the node services.

    Attributes:
        placement: Placement policy plugin (``membrane.placement``).
        route_threshold: Initial prefill-offload threshold in tokens; ``0``
            leaves offload decisions out of ``/route``.
        promote_replicas: Copies a hot fragment may reach; ``0`` disables
            promotion.
        dynamic_roles: Re-evaluate and advertise the node's role from load.
        origin: ``host:port`` of an origin this node caches for (read-through).
        require_compat: ``MODEL`` or ``MODEL:DTYPE``: refuse fragments
            stamped for another model; empty accepts any.
        prefix_cache_entries: Prefix lookups memoized.
        policy_interval_sec: Seconds between promotion and role passes.
    """

    placement: str = "ring"
    route_threshold: int = 0
    promote_replicas: int = 0
    dynamic_roles: bool = False
    origin: str = ""
    require_compat: str = ""
    prefix_cache_entries: int = 4096
    policy_interval_sec: float = 30.0

    def compat(self) -> ModelCompatibilityFingerprint | None:
        """The fingerprint written fragments must carry.

        Returns:
            ModelCompatibilityFingerprint | None: ``None`` when unchecked.
        """
        if not self.require_compat:
            return None
        model, _sep, dtype = self.require_compat.partition(":")
        return compat_hash(model, dtype=dtype or "float16")


class Services:
    """The memory API, routing, and background policies of one node.

    Attributes:
        options: The settings.
        memory: Reconstruction, prefix lookups, sessions, typed objects.
        view: What this node knows of the cluster.
        placement: ``POST /route``.
        promoter: Hot-fragment promotion (``None``: off).
        roles: Dynamic role policy (``None``: off).
        origin: Read-through to an origin (``None``: not a regional cache).
    """

    def __init__(
        self,
        options: ServiceOptions,
        node: Any,
        backend: Any = None,
        cluster: Any = None,
        replica_count: int = 2,
        queue_depth: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        """Build the services.

        Args:
            options: Settings.
            node: The local node.
            backend: The compute backend (prefill for reconstruction).
            cluster: The cluster manager, if any.
            replica_count: Copies per hash on the ring.
            queue_depth: Returns ``(waiting, saturated_depth)`` for the
                routing scheduler; zero load when ``None``.
        """
        from membrane.runtime.plugins import PLACEMENT

        self.options = options
        self.memory = MemoryService(
            node, backend, compat=options.compat(), prefix_cache_capacity=options.prefix_cache_entries
        )
        gpu_load = getattr(backend, "gpu_load", None)
        self.view = ClusterView(node, cluster, replica_count=replica_count, gpu_load=gpu_load)
        self.placement = PlacementService(
            self.view,
            PLACEMENT.get(options.placement)(),
            options.placement,
            memory=self.memory,
            route_threshold=options.route_threshold,
        )
        self.promoter = (
            Promoter(self.memory, self.view, options.promote_replicas)
            if options.promote_replicas > 0 and cluster is not None
            else None
        )
        self.roles = RolePolicy(self.view) if options.dynamic_roles else None
        self.origin = OriginLink(node, options.origin) if options.origin else None
        self.queue_depth = queue_depth
        interval = options.policy_interval_sec
        self.tasks: list[PeriodicTask] = []
        if self.promoter is not None:
            self.tasks.append(PeriodicTask("membrane-promotion", interval, self.promoter.run_once))
        if self.roles is not None:
            self.tasks.append(PeriodicTask("membrane-roles", interval, self.roles.run_once))
        if self.placement.scheduler is not None:
            self.tasks.append(PeriodicTask("membrane-route-adjust", ROUTE_ADJUST_INTERVAL_SEC, self.adjust_routing))
            self.tasks.append(PeriodicTask("membrane-route-refit", ROUTE_REFIT_INTERVAL_SEC, self.placement.reoptimize))

    @property
    def role(self) -> str:
        """The role this node advertises (``""`` without dynamic roles)."""
        return self.roles.role.value if self.roles is not None else ""

    def adjust_routing(self) -> int | None:
        """Feed the current queue depth to the routing scheduler.

        Returns:
            int | None: The effective offload threshold.
        """
        waiting, saturated = self.queue_depth() if self.queue_depth is not None else (0, 1)
        return self.placement.adjust(waiting, saturated)

    def start(self) -> None:
        """Start the periodic tasks."""
        for task in self.tasks:
            task.start()

    def stop(self, deadline_sec: float = 10.0) -> bool:
        """Stop the periodic tasks.

        Args:
            deadline_sec: Budget per task.

        Returns:
            bool: True when every task stopped in time.
        """
        return all([task.stop(deadline_sec) for task in self.tasks])


__all__ = ["ROUTE_ADJUST_INTERVAL_SEC", "ROUTE_REFIT_INTERVAL_SEC", "ServiceOptions", "Services"]
