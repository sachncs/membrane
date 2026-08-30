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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from membrane.compute.base import Backend
from membrane.gc import Sweeper, TombstoneTable
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
from membrane.registry import Registry
from membrane.snapshot import SNAPSHOT_SCHEMA_VERSION, ClusterEpochGuard, Snapshot
from membrane.transfer import TransferService
from membrane.transport.fastapi import FastAPIServer

logger = logging.getLogger(__name__)


# Registry mapping CLI/backend name strings to backend factories.
# Each factory receives (llm_url, llm_model, api_key) and returns
# a Backend instance. Backends whose optional dependencies are not
# installed fall back to RuntimeError at construction time when
# the CLI tries to instantiate them — same behavior as the
# previous inline string-dispatch code.
ComputeBackendFactory = Callable[[str, str, str], Backend]

COMPUTE_BACKENDS: dict[str, ComputeBackendFactory] = {}


def _register_compute_backends() -> None:
    """Populate the COMPUTE_BACKENDS registry from optional-backend imports."""
    from membrane.compute.cpu import CPU

    COMPUTE_BACKENDS["cpu"] = lambda _url, _model, _key: CPU()
    COMPUTE_BACKENDS["gpu"] = lambda _url, _model, _key: _try_import("GPU")()
    COMPUTE_BACKENDS["ollama"] = lambda url, model, _key: cast(
        Backend,
        _try_import("Ollama")(
            base_url=url or "http://localhost:11434",
            model=model or "llama3.2",
        ),
    )
    COMPUTE_BACKENDS["openai"] = lambda _url, model, key: cast(
        Backend,
        _try_import("OpenAI")(
            model=model or "gpt-4o-mini",
            api_key=key,
        ),
    )
    COMPUTE_BACKENDS["anthropic"] = lambda _url, model, key: cast(
        Backend,
        _try_import("Anthropic")(
            model=model or "claude-3-sonnet-20240229",
            api_key=key,
        ),
    )
    COMPUTE_BACKENDS["transformers"] = lambda _url, model, _key: cast(
        Backend,
        _try_import("Transformers")(
            model_id=model or "gpt2",
        ),
    )


def _try_import(class_name: str) -> Any:
    """Import an optional backend class by name.

    Returns the class object so callers can construct an
    instance with provider-specific kwargs (BaseURL, API
    key, model, etc.). The class is typed as ``Any`` here
    because each concrete provider's constructor accepts
    different arguments and Backend's own signature is the
    empty ``__init__(self)``.

    Args:
        class_name: Backend class to import (e.g., ``"Ollama"``).

    Returns:
        Any: The backend class (typed loosely so provider-
        specific constructors are accepted).

    Raises:
        RuntimeError: If the optional backend dependency is
            not installed.
    """
    from membrane.compute import anthropic as _anthropic
    from membrane.compute import gpu as _gpu
    from membrane.compute import ollama as _ollama
    from membrane.compute import openai as _openai
    from membrane.compute import transformers as _t

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
        state_dir: str | None = None,
        checkpoint_interval_sec: float = 30.0,
        cluster_epoch: int = 0,
        sweep_interval_sec: float = 30.0,
    ) -> None:
        """Initialize the server with all configured subsystems.

        Args:
            node: Local :class:`Node` instance.
            transport: ``"http"`` / ``"stdlib"`` / ``"grpc"``.
            compute: Backend name or pre-built instance.
            redis_url: Redis URL for the persistence layer.
            host: Bind address.
            port: Listen port.
            cluster_config: Optional cluster configuration.
            llm_url: LLM base URL.
            llm_model: LLM model identifier.
            api_key: LLM API key.
            state_dir: Optional directory under which
                ``{node_id}.json`` snapshots are persisted. When
                ``None``, no on-disk recovery is attempted and
                the server boots with empty membership / shard
                tables.
            checkpoint_interval_sec: How often the
                CheckpointThread writes a snapshot while the
                server is running. ``30`` matches the design
                plan.
            cluster_epoch: Live cluster epoch. Increment when
                the cluster topology changes in ways that should
                invalidate stale snapshots.
        """
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

        self.persistence = self.build_persistence(redis_url)

        self.cluster_manager: Cluster | None = None
        # GC plumbing: tombstones + periodic sweeper. Single
        # TombstoneTable is shared with the transport's
        # op_delete/op_tombstone so producers, peers, and the
        # sweeper converge on the same set. The block must run
        # before the Cluster constructor so the cluster can
        # share the table.
        self.tombstones = TombstoneTable()
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.sweeper: Sweeper | None = Sweeper(interval_sec=self.sweep_interval_sec)
        self.sweeper_thread: threading.Thread | None = None
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
                tombstones=self.tombstones,
            )
            self.transfer_service.cluster_manager = self.cluster_manager
            # Wire TransferService into the cluster so the migrator's
            # transfer_fn can push canonical bytes through the wire
            # path during shard migrations.
            self.cluster_manager.transfer_service = self.transfer_service

        self.transport = self.build_transport(transport, host, port)
        self.running = False
        self.thread: threading.Thread | None = None

        # Snapshotting and recovery plumbing. The Snapshot helper is
        # created lazily against the configured state_dir; the
        # epoch guard refuses to apply snapshots that fall more than
        # one step behind the live cluster epoch, so a node that
        # lost a long partition never rebuilds an obsolete map.
        self.state_dir = state_dir
        self.checkpoint_interval_sec = float(checkpoint_interval_sec)
        self.cluster_epoch = cluster_epoch
        self.snapshot: Snapshot | None = Snapshot(state_dir) if state_dir else None
        self.epoch_guard = ClusterEpochGuard(current=cluster_epoch)
        self.checkpoint_stop_event: threading.Event | None = None
        self.checkpoint_thread: threading.Thread | None = None

    def build_persistence(self, redis_url: str) -> Any:
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

    def build_transport(self, transport: str, host: str, port: int) -> Any:
        mtls = self.cluster_config.mtls if self.cluster_config is not None else None
        if transport == "http":
            return FastAPIServer(
                node=self.node,
                host=host,
                port=port,
                compute_backend=self.compute_backend,
                transfer_service=self.transfer_service,
                cluster_manager=self.cluster_manager,
                metrics_registry=self.metrics_registry,
                tls=mtls,
            )
        from membrane.transport.grpc import GrpcServer

        return GrpcServer(
            node=self.node,
            host=host,
            port=port,
            compute_backend=self.compute_backend,
            tls=mtls,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background thread."""
        # Re-hydrate durable state before any thread starts so the
        # cluster manager, snapshot, and transfer service are
        # populated atomically with respect to live traffic.
        self.restore_state()
        self.running = True
        if self.cluster_manager:
            self.cluster_manager.start()
        self.thread = threading.Thread(target=self.transport.start, daemon=True)
        self.thread.start()
        if self.cluster_manager is not None and self.checkpoint_interval_sec > 0:
            self.checkpoint_stop_event = threading.Event()
            self.checkpoint_thread = threading.Thread(
                target=self._checkpoint_loop,
                daemon=True,
                name="membrane-checkpoint",
            )
            self.checkpoint_thread.start()
        if self.sweeper is not None and self.sweep_interval_sec > 0:
            # The Sweeper depends on the cluster's directory and
            # tombstone table — only start it when those exist.
            directory = self.cluster_manager.directory if self.cluster_manager else None
            if directory is not None:
                registry_for_dir: Registry = directory

                def _forget(hashes: list[str]) -> None:
                    for h in hashes:
                        registry_for_dir.forget_fragment(h)

                self.sweeper.on_post_sweep = _forget
            self.sweeper_thread = threading.Thread(
                target=self._sweeper_loop,
                daemon=True,
                name="membrane-sweeper",
            )
            self.sweeper_thread.start()
        self.log_event("info", f"Server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the server gracefully."""
        if self.checkpoint_stop_event is not None:
            self.checkpoint_stop_event.set()
            if self.checkpoint_thread is not None:
                self.checkpoint_thread.join(timeout=2.0)
        # Flush a final checkpoint before tearing down so the next
        # process can rebuild from up-to-date state.
        self.checkpoint_state()
        self.running = False
        self.transport.stop()
        if self.cluster_manager:
            self.cluster_manager.stop()
        if self.sweeper is not None:
            self.sweeper.stop(timeout=2.0)
            self.sweeper_thread = None
        self.log_event("info", "Server stopped")

    def _checkpoint_loop(self) -> None:
        """Background loop writing snapshots every ``checkpoint_interval_sec``."""
        while self.running and self.checkpoint_stop_event is not None:
            if self.checkpoint_stop_event.wait(self.checkpoint_interval_sec):
                return
            try:
                self.checkpoint_state()
            except Exception as exc:  # pragma: no cover - background safety
                logger.warning("Checkpoint failed: %s", exc)

    def _sweeper_loop(self) -> None:
        """Background loop sweeping TTL + tombstones every ``sweep_interval_sec``.

        Uses :meth:`Node.evict` (TTL) for the eviction phase and
        the shared :class:`~membrane.gc.TombstoneTable` for the
        soft-delete sweep. The post-sweep observer forgets the
        directory entries of every hash touched.
        """
        node = self.node

        def _evict() -> list[str]:
            evicted: list[str] = node.evict(target_bytes=len(node.fragments) * 16)
            return evicted if evicted is not None else []

        while self.running:
            if not isinstance(self.sweeper, Sweeper):
                return
            try:
                self.sweeper.run_once(
                    evict_expired=_evict,
                    tombstones=self.tombstones,
                )
            except Exception as exc:  # pragma: no cover - background safety
                logger.warning("Sweep failed: %s", exc)
            # Sleep until next interval via the sweepEvent, not a
            # raw sleep, so stop() interrupts promptly.
            if self.sweeper.stop_event.wait(self.sweep_interval_sec):
                return

    def restore_state(self) -> None:
        """Re-hydrate membership / shard tables from the configured snapshot.

        No-op when ``state_dir`` was not provided at construction
        time. When the snapshot's cluster_epoch is more than one
        step behind the live epoch the persisted payload is
        discarded and the cluster starts fresh — a node coming
        back after a long partition must not rebuild an obsolete
        map. On a successful restore the persisted epoch is
        adopted as the new live epoch so the next checkpoint
        continues from there.
        """
        if self.snapshot is None:
            return
        payload = self.snapshot.load(self.node.node_id)
        if payload is None:
            return
        persisted_epoch = payload.get("cluster_epoch")
        if not self.epoch_guard.accept(persisted_epoch):
            logger.warning(
                "Snapshot for %s has epoch %s but live cluster is at %s; discarding",
                self.node.node_id,
                persisted_epoch,
                self.cluster_epoch,
            )
            self.snapshot.remove(self.node.node_id)
            return
        if self.cluster_manager is not None:
            self.cluster_manager.membership.load_snapshot(payload.get("membership", []))
            self.cluster_manager.shard_manager.load_snapshot(payload.get("shards", {}))
        server_section = payload.get("server", {})
        if server_section:
            self.request_count = int(server_section.get("request_count", 0))
            self.error_count = int(server_section.get("error_count", 0))
        # Adopt the persisted epoch as the live one so subsequent
        # checkpoints continue from that value rather than
        # resetting it back to the constructor's cluster_epoch.
        if persisted_epoch is not None:
            self.cluster_epoch = int(persisted_epoch)
            self.epoch_guard.current = max(int(persisted_epoch), self.epoch_guard.current)
        logger.info(
            "Restored snapshot for %s at epoch %s",
            self.node.node_id,
            persisted_epoch,
        )

    def checkpoint_state(self) -> None:
        """Persist the current membership / shard / counter state.

        No-op when ``state_dir`` was not provided. Bumps the live
        ``cluster_epoch`` so a subsequent restart writes a fresh
        value back.
        """
        if self.snapshot is None:
            return
        new_epoch = self.epoch_guard.bump()
        self.cluster_epoch = new_epoch
        server_section: dict[str, int] = {
            "request_count": self.request_count,
            "error_count": self.error_count,
        }
        shards_section: dict[str, object] = {}
        membership_section: list[dict[str, object]] = []
        if self.cluster_manager is not None:
            shards_section = self.cluster_manager.shard_manager.save_snapshot()
            membership_section = self.cluster_manager.membership.save_snapshot()
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "cluster_epoch": new_epoch,
            "captured_at": time.time(),
            "membership": membership_section,
            "shards": shards_section,
            "server": server_section,
        }
        self.snapshot.save(self.node.node_id, payload)

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
            redis_connected=self.persistence.ping(),
            load=self.node.heartbeat(),
        )

    def recent_events(self, n: int = 20) -> list[ServerEvent]:
        """Return the last ``n`` events."""
        return self.events[-n:]
