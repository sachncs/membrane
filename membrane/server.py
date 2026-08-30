"""Server: unified production server orchestrating transport, compute, and persistence.

Wraps an HTTP (stdlib or FastAPI) or gRPC transport, a compute
backend (CPU/GPU/Transformers/OpenAI/Anthropic/Ollama), and an
optional Redis persistence layer into a single runnable
service.

The server is also the entry point for the CLI's ``serve``
command and the TUI dashboard.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from membrane.compute.base import Backend
from membrane.metrics import (
    ClusterMetrics,
    MetricsCollector,
    NodeMetrics,
    PersistenceMetrics,
    TransportMetrics,
)
from membrane.network.cluster import Cluster
from membrane.network.config import ClusterConfig
from membrane.node import Node
from membrane.persistence.memory import Memory
from membrane.persistence.redis import Redis
from membrane.transfer import TransferService
from membrane.transport.fastapi import FastAPIServer
from membrane.transport.http import HTTPServer

logger = logging.getLogger(__name__)


# Registry mapping CLI/backend name strings to backend factories.
# Each factory receives (llm_url, llm_model, api_key) and returns
# a Backend instance. Backends whose optional dependencies are not
# installed fall back to RuntimeError at construction time when
# the CLI tries to instantiate them — same behavior as the
# previous inline string-dispatch code.
COMPUTE_BACKENDS: dict[str, Any] = {}


def _register_compute_backends() -> None:
    """Populate the COMPUTE_BACKENDS registry from optional-backend imports."""
    from membrane.compute.cpu import CPU

    COMPUTE_BACKENDS["cpu"] = lambda _url, _model, _key: CPU()
    COMPUTE_BACKENDS["gpu"] = lambda _url, _model, _key: _try_import("GPU")()
    COMPUTE_BACKENDS["ollama"] = lambda url, model, _key: _try_import("Ollama")(
        base_url=url or "http://localhost:11434",
        model=model or "llama3.2",
    )
    COMPUTE_BACKENDS["openai"] = lambda _url, model, key: _try_import("OpenAI")(
        model=model or "gpt-4o-mini",
        api_key=key,
    )
    COMPUTE_BACKENDS["anthropic"] = lambda _url, model, key: _try_import("Anthropic")(
        model=model or "claude-3-sonnet-20240229",
        api_key=key,
    )
    COMPUTE_BACKENDS["transformers"] = lambda _url, model, _key: _try_import("Transformers")(
        model_id=model or "gpt2",
    )


def _try_import(class_name: str) -> type[Backend]:
    """Import an optional backend class by name.

    Args:
        class_name: Backend class to import (e.g., ``"Ollama"``).

    Returns:
        The backend class.

    Raises:
        RuntimeError: If the optional backend dependency is
            not installed.
    """
    from membrane.compute import anthropic as _anthropic  # noqa: F401
    from membrane.compute import gpu as _gpu  # noqa: F401
    from membrane.compute import ollama as _ollama  # noqa: F401
    from membrane.compute import openai as _openai  # noqa: F401
    from membrane.compute import transformers as _t  # noqa: F401

    module_map = {
        "Ollama": _ollama,
        "OpenAI": _openai,
        "Anthropic": _anthropic,
        "GPU": _gpu,
        "Transformers": _t,
    }
    module = module_map[class_name]
    return getattr(module, class_name)


_register_compute_backends()


@dataclass
class ServerEvent:
    """A single server event for dashboard logging.

    Attributes:
        timestamp: Unix time at which the event was recorded.
        level: Log level (``"info"``, ``"warn"``, ``"error"``,
            etc.).
        message: Human-readable description.
        node_id: Optional node identifier associated with the
            event.
        bytes_affected: Optional size in bytes (e.g., a
            transfer size).
    """

    timestamp: float
    level: str
    message: str
    node_id: str = ""
    bytes_affected: int = 0


@dataclass
class ServerDiagnostics:
    """Snapshot of server health and performance.

    Attributes:
        node_id: Identifier of the local node.
        uptime_seconds: Seconds since :meth:`start`.
        memory_used_bytes: Bytes currently held by the node.
        memory_limit_bytes: Configured node memory cap.
        fragment_count: Number of fragments stored locally.
        primary_count: Number of fragments owned as primary.
        hit_rate: External cache hit rate (currently always
            ``0.0``; tracked outside the server).
        miss_rate: External cache miss rate.
        request_count: Cumulative request count.
        error_count: Cumulative error count.
        connected_nodes: Number of distinct peers seen.
        backend_name: Compute backend descriptor.
        redis_connected: True when the Redis backend is
            reachable.
        load: Local node load ratio.
    """

    node_id: str
    uptime_seconds: float
    memory_used_bytes: int
    memory_limit_bytes: int
    fragment_count: int
    primary_count: int
    hit_rate: float
    miss_rate: float
    request_count: int
    error_count: int
    connected_nodes: int
    backend_name: str
    redis_connected: bool
    load: float


class Server:
    """Unified production server for Membrane.

    Args:
        node: Node instance.
        transport: ``"http"`` (FastAPI), ``"stdlib"`` (stdlib
            HTTP), or ``"grpc"``.
        compute: ``"cpu"``, ``"gpu"``, ``"ollama"``,
            ``"openai"``, ``"anthropic"``, or ``"transformers"``.
            Alternatively an existing :class:`Backend` instance.
        redis_url: Redis URL, or ``""`` to disable persistence.
        host: Bind address.
        port: Listen port.
        cluster_config: Optional cluster configuration for
            peer-to-peer mode.
        llm_url: Base URL for the chosen LLM backend.
        llm_model: Model identifier for the chosen backend.
        api_key: API key for the chosen backend.
    """

    def __init__(
        self,
        node: Node,
        transport: str = "http",
        compute: str | Backend = "cpu",
        redis_url: str = "",
        host: str = "0.0.0.0",
        port: int = 8080,
        cluster_config: ClusterConfig | None = None,
        llm_url: str = "",
        llm_model: str = "",
        api_key: str = "",
    ) -> None:
        """Initialize the server with all configured subsystems."""
        self.node = node
        self.transport_type = transport
        self.redis_url = redis_url
        self.host = host
        self.port = port
        self.cluster_config = cluster_config

        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.events: list[ServerEvent] = []
        self.connected_nodes: set[str] = set()

        self.metrics_registry = MetricsCollector()
        self.metrics_transport = TransportMetrics(self.metrics_registry)
        self.metrics_cluster = ClusterMetrics(self.metrics_registry)
        self.metrics_persistence = PersistenceMetrics(self.metrics_registry)
        self.metrics_node = NodeMetrics(self.metrics_registry)

        if isinstance(compute, Backend):
            self.compute_backend = compute
            self.compute_type = compute.device_name()
        else:
            self.compute_type = compute
            factory = COMPUTE_BACKENDS.get(compute)
            if factory is None:
                raise ValueError(
                    f"Unknown compute backend '{compute}'. "
                    f"Available: {sorted(COMPUTE_BACKENDS)}"
                )
            self.compute_backend = factory(llm_url, llm_model, api_key)

        self.persistence = self._build_persistence(redis_url)

        self.cluster_manager: Cluster | None = None
        self.transfer_service = TransferService(
            cluster_manager=self.cluster_manager,
            local_node=self.node,
        )
        if cluster_config is not None:
            self.cluster_manager = Cluster(
                node_id=self.node.node_id,
                host=host,
                port=port,
                node=self.node,
                config=cluster_config,
            )
            self.transfer_service.cluster_manager = self.cluster_manager

        self.transport = self._build_transport(transport, host, port)
        self.running = False
        self.thread: threading.Thread | None = None

    def _build_persistence(self, redis_url: str) -> Any:
        from membrane.persistence.cache import CachingPersistence

        backend: Any = Memory()
        if redis_url:
            try:
                redis_backend = Redis(redis_url)
                if redis_backend.ping():
                    backend = redis_backend
                    logger.info("Redis connected at %s", redis_url)
                else:
                    logger.warning("Redis at %s unreachable; using in-memory persistence", redis_url)
            except Exception as exc:
                logger.warning("Redis connection failed (%s); using in-memory persistence", exc)
        return CachingPersistence(backend)

    def _build_transport(self, transport: str, host: str, port: int) -> Any:
        if transport == "http":
            return FastAPIServer(
                node=self.node,
                host=host,
                port=port,
                compute_backend=self.compute_backend,
                transfer_service=self.transfer_service,
                cluster_manager=self.cluster_manager,
                metrics_registry=self.metrics_registry,
            )
        if transport == "stdlib":
            return HTTPServer(
                node=self.node,
                host=host,
                port=port,
                compute_backend=self.compute_backend,
                transfer_service=self.transfer_service,
                cluster_manager=self.cluster_manager,
            )
        from membrane.transport.grpc import GrpcServer

        return GrpcServer(
            node=self.node,
            host=host,
            port=port,
            compute_backend=self.compute_backend,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background thread."""
        self.running = True
        if self.cluster_manager:
            self.cluster_manager.start()
        self.thread = threading.Thread(target=self.transport.start, daemon=True)
        self.thread.start()
        self.log_event("info", f"Server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the server gracefully."""
        self.running = False
        self.transport.stop()
        if self.cluster_manager:
            self.cluster_manager.stop()
        self.log_event("info", "Server stopped")

    def join(self) -> None:
        """Block until the server thread exits."""
        if self.thread:
            self.thread.join()

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        level: str,
        message: str,
        node_id: str = "",
        bytes_affected: int = 0,
    ) -> None:
        """Record a server event.

        Events are stored in a bounded buffer (the most recent
        10,000 events are kept; older entries are trimmed to
        the most recent 5,000).
        """
        event = ServerEvent(
            timestamp=time.time(),
            level=level,
            message=message,
            node_id=node_id,
            bytes_affected=bytes_affected,
        )
        self.events.append(event)
        if len(self.events) > 10_000:
            self.events = self.events[-5_000:]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> ServerDiagnostics:
        """Return a current snapshot of server health."""
        stats = self.node.get_stats()
        now = time.time()
        connected = len(self.connected_nodes)
        if self.cluster_manager:
            connected = max(connected, len(self.cluster_manager.membership.to_json()))
        return ServerDiagnostics(
            node_id=self.node.node_id,
            uptime_seconds=now - self.start_time,
            memory_used_bytes=stats.memory_used_bytes,
            memory_limit_bytes=stats.memory_limit_bytes,
            fragment_count=stats.fragment_count,
            primary_count=stats.primary_count,
            hit_rate=0.0,
            miss_rate=0.0,
            request_count=self.request_count,
            error_count=self.error_count,
            connected_nodes=connected,
            backend_name=self.compute_backend.device_name(),
            redis_connected=isinstance(self.persistence, Redis) and self.persistence.ping(),
            load=self.node.heartbeat(),
        )

    def recent_events(self, n: int = 20) -> list[ServerEvent]:
        """Return the last ``n`` events."""
        return self.events[-n:]
