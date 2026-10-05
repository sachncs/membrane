"""Typed access to the objects a FastAPI app carries in ``app.state``.

Handlers used to read dependencies with ``getattr(app.state, "name",
None)`` scattered across modules: untyped, with defaults repeated at
each call site. :class:`AppContext` is the one place that knows every
name and its default. ``create_app`` and :class:`~membrane.server.Server`
still populate ``app.state`` (so tests can swap in stubs), and handlers
read through :func:`app_context`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

if TYPE_CHECKING:
    from membrane.audit import AuditLog
    from membrane.auth import Authenticator
    from membrane.compute.base import Backend
    from membrane.gc import TombstoneTable
    from membrane.metrics import ClusterMetrics, MetricsCollector, TransportMetrics
    from membrane.network.cluster import Cluster
    from membrane.node import Node
    from membrane.transfer import TransferService
    from membrane.transport.limits import TransportLimits


class AppContext:
    """Typed, read-only view of one app's ``app.state``.

    Attributes:
        state: The underlying Starlette ``State`` object.
    """

    __slots__ = ("state",)

    def __init__(self, app: FastAPI) -> None:
        """Wrap ``app``'s state.

        Args:
            app: The FastAPI application.
        """
        self.state = app.state

    def __get(self, name: str) -> Any:
        """Read ``name`` from the state, or ``None`` when unset.

        Args:
            name: Attribute name.

        Returns:
            Any: The attribute value, or ``None``.
        """
        return getattr(self.state, name, None)

    @property
    def node(self) -> Node | None:
        """The local node."""
        return self.__get("node")

    @property
    def cluster(self) -> Cluster | None:
        """The cluster manager, when the node is part of a cluster."""
        return self.__get("cluster_manager")

    @property
    def server(self) -> Any:
        """The owning :class:`~membrane.server.Server`, when there is one."""
        return self.__get("server")

    @property
    def draining(self) -> bool:
        """Whether the owning server is draining."""
        return bool(getattr(self.server, "is_draining", False))

    @property
    def quorum_attempt(self) -> Callable[..., Any] | None:
        """The quorum fan-out used by strong and quorum writes."""
        return self.__get("quorum_attempt")

    @property
    def authenticator(self) -> Authenticator | None:
        """The inbound authenticator, when authentication is on."""
        return self.__get("authenticator")

    @property
    def compute_backend(self) -> Backend | None:
        """The compute backend used by prefill."""
        return self.__get("compute_backend")

    @property
    def transfer_service(self) -> TransferService | None:
        """The transfer service used by sync."""
        return self.__get("transfer_service")

    @property
    def metrics_registry(self) -> MetricsCollector | None:
        """The Prometheus metrics registry."""
        return self.__get("metrics_registry")

    @property
    def transport_metrics(self) -> TransportMetrics | None:
        """Per-endpoint request metrics."""
        return self.__get("transport_metrics")

    @property
    def cluster_metrics(self) -> ClusterMetrics | None:
        """Cluster and per-tenant metrics."""
        return self.__get("cluster_metrics")

    @property
    def refresh_metrics(self) -> Callable[[], None] | None:
        """Hook that refreshes point-in-time gauges before a scrape."""
        return self.__get("refresh_metrics")

    @property
    def tombstones(self) -> TombstoneTable | None:
        """The shared tombstone table."""
        return self.__get("tombstones")

    @property
    def limits(self) -> TransportLimits | None:
        """HTTP capacity settings."""
        return self.__get("limits")

    @property
    def audit_log(self) -> AuditLog | None:
        """The admin audit log, once created."""
        return self.__get("audit_log")


def app_context(app: FastAPI) -> AppContext:
    """Return the typed context of ``app``.

    Args:
        app: The FastAPI application.

    Returns:
        AppContext: A view over ``app.state``.
    """
    return AppContext(app)


__all__ = ["AppContext", "app_context"]
