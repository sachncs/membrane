"""MembraneServer: unified production server orchestrating transport, compute, and persistence.

Wraps an HTTP (stdlib or FastAPI) or gRPC transport, a compute
backend (CPU/GPU/Transformers/OpenAI/Anthropic/Ollama), and an
optional Redis persistence layer into a single runnable
service.

The server is also the entry point for the CLI's ``serve``
command and the TUI dashboard. It owns:

* A :class:`MembraneNode` instance.
* A :class:`ComputeBackend`.
* A persistence backend (:class:`InMemoryBackend` or
  :class:`RedisBackend`).
* An optional :class:`ClusterManager` and matching
  :class:`RemoteTransferService`.
* An in-memory event log surfaced via
  :meth:`recent_events` for the dashboard.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from membrane.compute.backend import ComputeBackend
from membrane.compute.cpu_backend import CPUBackend
from membrane.compute.gpu_backend import GPUBackend
from membrane.membrane_node import MembraneNode
from membrane.network.cluster_manager import ClusterManager
from membrane.network.config import ClusterConfig
from membrane.network.remote_transfer import RemoteTransferService
from membrane.persistence.memory_backend import InMemoryBackend
from membrane.persistence.redis_backend import RedisBackend
from membrane.transport.fastapi_server import FastAPIServer
from membrane.transport.http_server import HTTPServer

logger = logging.getLogger(__name__)


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


class MembraneServer:
    """Unified production server for Membrane.

    Args:
        node: MembraneNode instance.
        transport: ``"http"`` (FastAPI), ``"stdlib"`` (stdlib
            HTTP), or ``"grpc"``.
        compute: ``"cpu"``, ``"gpu"``, ``"ollama"``,
            ``"openai"``, ``"anthropic"``, or ``"transformers"``.
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
        node: MembraneNode,
        transport: str = "http",
        compute: str = "cpu",
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
        self.compute_type = compute
        self.redis_url = redis_url
        self.host = host
        self.port = port
        self.cluster_config = cluster_config

        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.events: list[ServerEvent] = []
        self.connected_nodes: set[str] = set()

        # Compute backend.
        self.compute_backend = self._make_compute_backend(compute, llm_url, llm_model, api_key)

        # Persistence.
        self._setup_persistence(redis_url)

        # Cluster.
        self._setup_cluster(cluster_config, host, port)

        # Transport.
        self._setup_transport(transport, host, port)

    def _make_compute_backend(
        self,
        compute: str,
        llm_url: str,
        llm_model: str,
        api_key: str,
    ) -> ComputeBackend:
        """Construct the compute backend matching ``compute``.

        Args:
            compute: Backend name.
            llm_url: Base URL (used for Ollama).
            llm_model: Model identifier.
            api_key: API key (used for OpenAI / Anthropic).

        Returns:
            ComputeBackend: The constructed backend instance.
        """
        if compute == "gpu":
            return GPUBackend()
        if compute == "ollama":
            from membrane.compute.ollama_backend import OllamaBackend
            url = llm_url or "http://localhost:11434"
            model = llm_model or "llama3.2"
            return OllamaBackend(base_url=url, model=model)
        if compute == "openai":
            from membrane.compute.openai_backend import OpenAIBackend
            model = llm_model or "gpt-4o-mini"
            return OpenAIBackend(api_key=api_key, model=model)
        if compute == "anthropic":
            from membrane.compute.anthropic_backend import AnthropicBackend
            model = llm_model or "claude-3-sonnet-20240229"
            return AnthropicBackend(api_key=api_key, model=model)
        if compute == "transformers":
            from membrane.compute.transformers_backend import TransformersBackend
            model = llm_model or "gpt2"
            return TransformersBackend(model_id=model)
        # Default: CPU backend.
        return CPUBackend()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_persistence(self, redis_url: str) -> None:
        """Initialize the persistence backend.

        When ``redis_url`` is set and reachable, use Redis;
        otherwise fall back to the in-memory backend.
        """
        self.persistence: Any = InMemoryBackend()
        if redis_url:
            try:
                redis_backend = RedisBackend(redis_url)
                if redis_backend.ping():
                    self.persistence = redis_backend
                    logger.info("Redis connected at %s", redis_url)
                else:
                    logger.warning(
                        "Redis at %s unreachable; using in-memory persistence", redis_url
                    )
            except Exception as exc:
                logger.warning(
                    "Redis connection failed (%s); using in-memory persistence", exc
                )

    def _setup_cluster(
        self,
        cluster_config: ClusterConfig | None,
        host: str,
        port: int,
    ) -> None:
        """Initialize the cluster manager and transfer service."""
        self.cluster_manager: ClusterManager | None = None
        self.transfer_service = RemoteTransferService(
            cluster_manager=self.cluster_manager,
            local_node=self.node,
        )
        if cluster_config is not None:
            self.cluster_manager = ClusterManager(
                node_id=self.node.node_id,
                host=host,
                port=port,
                node=self.node,
                config=cluster_config,
            )
            # Re-bind the transfer service to the now-available
            # cluster manager.
            self.transfer_service.cluster_manager = self.cluster_manager

    def _setup_transport(
        self,
        transport: str,
        host: str,
        port: int,
    ) -> None:
        """Initialize the wire-protocol transport.

        Args:
            transport: ``"http"``, ``"stdlib"``, or ``"grpc"``.
            host: Bind host.
            port: Listen port.
        """
        self.transport: Any
        if transport == "http":
            self.transport = FastAPIServer(
                node=self.node,
                host=host,
                port=port,
                compute_backend=self.compute_backend,
                transfer_service=self.transfer_service,
                cluster_manager=self.cluster_manager,
            )
        elif transport == "stdlib":
            self.transport = HTTPServer(
                node=self.node,
                host=host,
                port=port,
                compute_backend=self.compute_backend,
                transfer_service=self.transfer_service,
                cluster_manager=self.cluster_manager,
            )
        else:
            from membrane.transport.grpc_server import GrpcServer

            self.transport = GrpcServer(
                node=self.node,
                host=host,
                port=port,
                compute_backend=self.compute_backend,
            )

        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background thread.

        Also starts the :class:`ClusterManager` (if configured)
        in its own background threads.
        """
        self._running = True
        if self.cluster_manager:
            self.cluster_manager.start()
        self._thread = threading.Thread(target=self.transport.start, daemon=True)
        self._thread.start()
        self.log_event("info", f"Server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the server gracefully.

        Stops the transport and (when configured) the cluster
        manager. The background thread exits shortly thereafter.
        """
        self._running = False
        self.transport.stop()
        if self.cluster_manager:
            self.cluster_manager.stop()
        self.log_event("info", "Server stopped")

    def join(self) -> None:
        """Block until the server thread exits."""
        if self._thread:
            self._thread.join()

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

        Args:
            level: Log level.
            message: Human-readable description.
            node_id: Optional node identifier.
            bytes_affected: Optional size in bytes.
        """
        event = ServerEvent(
            timestamp=time.time(),
            level=level,
            message=message,
            node_id=node_id,
            bytes_affected=bytes_affected,
        )
        self.events.append(event)
        # Keep the last 10,000 events. When the buffer exceeds
        # that size, trim to the most recent 5,000 to bound
        # memory usage without losing recent context.
        if len(self.events) > 10_000:
            self.events = self.events[-5_000:]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> ServerDiagnostics:
        """Return a current snapshot of server health.

        Returns:
            ServerDiagnostics: Snapshot suitable for the TUI
            dashboard or external monitoring.
        """
        stats = self.node.get_stats()
        now = time.time()
        connected = len(self.connected_nodes)
        if self.cluster_manager:
            # Prefer the cluster manager's count when available;
            # it's the authoritative source of membership.
            connected = max(connected, len(self.cluster_manager.get_peers()))
        return ServerDiagnostics(
            node_id=self.node.node_id,
            uptime_seconds=now - self.start_time,
            memory_used_bytes=stats.memory_used_bytes,
            memory_limit_bytes=stats.memory_limit_bytes,
            fragment_count=stats.fragment_count,
            primary_count=stats.primary_count,
            # Hit/miss rates are tracked externally by the cache
            # manager; the server exposes zeros here.
            hit_rate=0.0,
            miss_rate=0.0,
            request_count=self.request_count,
            error_count=self.error_count,
            connected_nodes=connected,
            backend_name=self.compute_backend.device_name(),
            redis_connected=isinstance(self.persistence, RedisBackend)
            and self.persistence.ping(),
            load=self.node.heartbeat(),
        )

    def recent_events(self, n: int = 20) -> list[ServerEvent]:
        """Return the last ``n`` events.

        Args:
            n: Maximum number of events to return. Values
                larger than the buffer length return the whole
                buffer.

        Returns:
            list[ServerEvent]: Newest events first when
            ``n`` is negative; otherwise the tail of the
            buffer.
        """
        return self.events[-n:]

    # ------------------------------------------------------------------
    # Peer tracking
    # ------------------------------------------------------------------

    def register_peer(self, node_id: str) -> None:
        """Register a connected peer node.

        If a cluster manager is configured, the peer is also
        added to its membership table (using the host/port
        resolved from the cluster manager's view).

        Args:
            node_id: Identifier of the new peer.
        """
        self.connected_nodes.add(node_id)
        if self.cluster_manager:
            # Resolve peer info from cluster manager and
            # forward to add_peer so host/port are known.
            peers = self.cluster_manager.get_peers()
            for p in peers:
                if p.get("node_id") == node_id:
                    self.cluster_manager.add_peer(node_id, p["host"], p["port"])
                    break
        self.log_event("info", f"Peer connected: {node_id}")

    def unregister_peer(self, node_id: str) -> None:
        """Unregister a disconnected peer node.

        Args:
            node_id: Identifier of the departing peer.
        """
        self.connected_nodes.discard(node_id)
        if self.cluster_manager:
            self.cluster_manager.remove_peer(node_id)
        self.log_event("warn", f"Peer disconnected: {node_id}")
