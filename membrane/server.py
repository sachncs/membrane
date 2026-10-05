"""Server: unified production server orchestrating transport, compute, and persistence.

:class:`Server` wires the components built in :mod:`membrane.runtime`
into one runnable service: the FastAPI HTTP transport, a compute
backend chosen from the plugin registry, write-behind Redis
persistence, the cluster manager, and the checkpoint / sweeper tasks.

It is the entry point for the CLI's ``serve`` command and the TUI
dashboard. The builders it uses, and the dashboard's event and
diagnostics types, are re-exported here for compatibility.
"""

import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Any

from membrane.audit import AuditLog, FileAuditStorage
from membrane.auth import Authenticator
from membrane.cache_metrics import CacheMetrics
from membrane.canonical import set_default_registry
from membrane.compute.base import Backend
from membrane.gc import Sweeper, TombstoneTable
from membrane.integrity import MerkleDrift, record_merkle_drift
from membrane.metrics import (
    ClusterMetrics,
    MetricsCollector,
    NodeMetrics,
    PeerHealthMetrics,
    PersistenceMetrics,
    TransportMetrics,
)
from membrane.network.cluster import Cluster
from membrane.network.config import ClusterConfig
from membrane.network.lag import record_replication_lag
from membrane.network.peer import get_default_peer_credentials
from membrane.node import Node
from membrane.otel_tracer.otel import TRACING
from membrane.quorum import QuorumReplicator
from membrane.registry import Registry
from membrane.runtime.components import (
    build_authenticator,
    build_content_store,
    build_persistence,
    configure_peer_access,
    is_durable,
    resolves_to_loopback,
)
from membrane.runtime.events import (
    DrainFinished,
    DrainStarted,
    EventBus,
    FragmentRemoved,
    FragmentStored,
    PeerJoined,
    PeerLeft,
)
from membrane.runtime.lifecycle import PeriodicTask
from membrane.runtime.observability import EventLog, ServerDiagnostics, ServerEvent
from membrane.runtime.persistence_writer import PersistenceWriter
from membrane.runtime.plugins import COMPUTE_BACKENDS, HOOKS
from membrane.security.keyring import DirectoryKeyring
from membrane.snapshot import SNAPSHOT_SCHEMA_VERSION, ClusterEpochGuard, Snapshot
from membrane.transfer import TransferService
from membrane.transport.acme import ACMEConfig, ensure_certificate
from membrane.transport.fastapi import FastAPIServer
from membrane.transport.limits import TransportLimits
from membrane.transport.spiffe import REFRESH_INTERVAL_SEC, SPIFFEClient, SPIFFEConfig
from membrane.transport.tls import MTLSConfig
from membrane.transport.tls_rotation import CertRotationWatcher, cert_not_after

logger = logging.getLogger(__name__)


class Server:
    """Unified production server for Membrane.

    Args:
        node: Node instance.
        transport: ``"http"`` (FastAPI); the only supported value.
        compute: A name from the compute plugin registry (built-ins:
            ``cpu``, ``gpu``, ``ollama``, ``openai``, ``anthropic``,
            ``transformers``; see :mod:`membrane.runtime.plugins`), or
            an existing :class:`Backend` instance.
        redis_url: Redis URL, or ``""`` to disable persistence.
        host: Bind address.
        port: Listen port.
        cluster_config: Optional cluster configuration for
            peer-to-peer mode.
        llm_url: Base URL for the chosen LLM backend.
        llm_model: Model identifier for the chosen backend.
        api_key: API key for the chosen backend.
        authenticator: Inbound request authenticator. When ``None``
            and ``cluster_config.mtls`` requires client certificates,
            an :class:`~membrane.auth.mtls.MTLSAuthenticator` is built
            from it; otherwise the node serves unauthenticated.
        peer_api_key: Bearer token this node presents to its peers.
            Required in a multi-node cluster that authenticates with
            API keys; the key needs ``admin`` scope because peers
            replicate every tenant's fragments and propagate deletes.
        peer_networks: CIDR ranges the cluster's peers live in.
            Outbound peer calls to these ranges bypass the SSRF
            private-address blocklist.
        tls: mTLS configuration for the listener and for peer
            calls. Defaults to ``cluster_config.mtls``.
        limits: HTTP capacity settings (concurrency bound, rate limit,
            connections, API docs).
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
        authenticator: Authenticator | None = None,
        peer_api_key: str = "",
        peer_networks: tuple[str, ...] = (),
        tls: MTLSConfig | None = None,
        limits: TransportLimits | None = None,
        persistence: str = "",
        load_hooks: bool = True,
        tls_files: tuple[str, str] | None = None,
        acme: ACMEConfig | None = None,
        spiffe: SPIFFEConfig | None = None,
        audit_path: str | None = None,
    ) -> None:
        """Initialize the server with all configured subsystems.

        Args:
            node: Local :class:`Node` instance.
            transport: ``"http"``.
            compute: Compute plugin name or pre-built instance.
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
            sweep_interval_sec: Seconds between TTL / tombstone sweeps.
            authenticator: Optional :class:`~membrane.auth.Authenticator`. When
                set, every route except ``/livez`` and ``/readyz`` authenticates
                the caller and enforces its route scope.
            peer_api_key: Bearer key this node presents to its peers (needs the
                ``admin`` scope).
            peer_networks: CIDR ranges of the peer network, exempt from the SSRF
                private-address block.
            tls: mTLS configuration; defaults to ``cluster_config.mtls``.
            limits: HTTP capacity settings; defaults to
                :class:`~membrane.transport.limits.TransportLimits`.
            persistence: Persistence plugin name (``membrane.persistence``);
                defaults to ``redis`` with a URL, else ``memory``.
            load_hooks: Run every installed ``membrane.hooks`` entry point.
            tls_files: ``(cert_path, key_path)`` to watch; a changed pair is
                served on new connections without a restart (also on SIGHUP).
            acme: Renew the ACME certificate in ``tls_files`` twice a day.
            spiffe: Re-fetch the SVID from the Workload API and serve a
                renewed one without a restart.
            audit_path: Persist the admin audit log here (JSON Lines); in
                memory only when ``None``.
        """
        self.node = node
        self.limits = limits or TransportLimits()
        self.transport_type = transport
        self.redis_url = redis_url
        self.host = host
        self.port = port
        self.cluster_config = cluster_config
        self.tls = tls or (cluster_config.mtls if cluster_config is not None else None)
        self.authenticator = authenticator or build_authenticator(self.tls)
        if cluster_config is not None:
            configure_peer_access(cluster_config, self.tls, peer_api_key, peer_networks)

        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.events = EventLog()
        self.connected_nodes: set[str] = set()

        self.metrics_registry = MetricsCollector()
        self.metrics_transport = TransportMetrics(self.metrics_registry)
        self.metrics_cluster = ClusterMetrics(self.metrics_registry)
        self.metrics_persistence = PersistenceMetrics(self.metrics_registry)
        self.metrics_node = NodeMetrics(self.metrics_registry)
        self.metrics_peers = PeerHealthMetrics(self.metrics_registry)
        # Corrupt canonical frames are counted wherever they are parsed.
        set_default_registry(self.metrics_registry)
        self.node.set_eviction_counter(lambda reason, count: self.metrics_node.evictions.inc(count, reason=reason))

        if isinstance(compute, Backend):
            self.compute_backend = compute
            self.compute_type = compute.device_name()
        else:
            self.compute_type = compute
            self.compute_backend = COMPUTE_BACKENDS.get(compute)(llm_url, llm_model, api_key)

        # Redis writes happen on a background thread so the node's lock
        # never waits on a network round trip.
        self.persistence = build_persistence(redis_url, persistence)
        self.durable = is_durable(self.persistence)
        self.persistence_writer = PersistenceWriter(self.persistence, node.node_id, self.metrics_persistence)
        # Node changes fan out to persistence and to event subscribers.
        self.event_bus = EventBus()
        self.node.set_persistence_hooks(self.on_fragment_stored, self.on_fragment_removed)

        # Tombstones are shared by the transport's delete path, the
        # cluster, and the sweeper, so all three converge on one set.
        self.tombstones = TombstoneTable()
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.sweeper = Sweeper(interval_sec=self.sweep_interval_sec)
        self.cluster_manager: Cluster | None = None
        self.replicator: QuorumReplicator | None = None
        self.transfer_service = TransferService(cluster_manager=None, local_node=self.node)
        if cluster_config is not None:
            self.cluster_manager = Cluster(
                node_id=self.node.node_id,
                host=host,
                port=port,
                node=self.node,
                config=cluster_config,
                tombstones=self.tombstones,
            )
            # The migrator pushes canonical bytes through the transfer
            # service during shard migrations.
            self.transfer_service.cluster_manager = self.cluster_manager
            self.cluster_manager.transfer_service = self.transfer_service
            self.replicator = QuorumReplicator()

        if self.cluster_manager is not None:
            self.cluster_manager.membership.listeners.append(self.on_membership_change)

        self.transport = self.build_transport(transport, host, port)
        self.audit_log = AuditLog.open(FileAuditStorage(Path(audit_path))) if audit_path else AuditLog()
        self.transport.app.state.audit_log = self.audit_log
        self.metrics_peers.audit_chain_valid.set(0.0 if self.audit_log.broken_at is not None else 1.0)
        self.running = False
        self.thread: threading.Thread | None = None
        # While draining, ``/readyz`` and writes return 503 so load
        # balancers and clients move to other nodes.
        self.is_draining = False
        self.__stopped = False

        # Snapshots: the epoch guard refuses snapshots more than one
        # step behind the live cluster epoch, so a node back from a
        # long partition never rebuilds an obsolete map.
        self.state_dir = state_dir
        self.checkpoint_interval_sec = float(checkpoint_interval_sec)
        self.cluster_epoch = cluster_epoch
        self.snapshot: Snapshot | None = Snapshot(state_dir) if state_dir else None
        self.epoch_guard = ClusterEpochGuard(current=cluster_epoch)
        self.checkpoint_task = PeriodicTask("membrane-checkpoint", self.checkpoint_interval_sec, self.checkpoint_state)
        self.sweeper_task = PeriodicTask("membrane-sweeper", self.sweep_interval_sec, self.sweep_once)
        # A directory of versioned data keys is re-read every minute; a
        # new version re-encrypts existing blobs under it.
        self.key_refresh_task = PeriodicTask("membrane-key-refresh", 60.0, self.refresh_data_keys)
        self.acme = acme
        self.spiffe = spiffe
        self.spiffe_task = PeriodicTask("membrane-spiffe-refresh", REFRESH_INTERVAL_SEC, self.refresh_svid)
        self.acme_task = PeriodicTask("membrane-acme-renewal", 12 * 3600.0, self.renew_acme_certificate)
        self.cert_watcher = (
            CertRotationWatcher(cert_path=tls_files[0], key_path=tls_files[1], on_rotate=self.rotate_tls)
            if tls_files is not None and self.tls is not None
            else None
        )
        if load_hooks:
            self.run_hooks()

    def run_hooks(self) -> None:
        """Call every installed ``membrane.hooks`` factory with ``(event_bus, self)``."""
        for name, factory in HOOKS.load_all().items():
            try:
                factory(self.event_bus, self)
                logger.info("hook %s installed", name)
            except Exception:
                logger.exception("hook %s failed to install", name)

    def on_fragment_stored(self, fragment: Any, is_primary: bool) -> None:
        """Persist and announce a newly stored fragment (node hook; runs under the node lock).

        Args:
            fragment: The fragment.
            is_primary: Whether this node owns the primary copy.
        """
        if self.durable:
            self.persistence_writer.store(fragment, is_primary)
        self.event_bus.publish(FragmentStored(fragment.identity.payload_hash, fragment.tenant_id, is_primary))

    def on_fragment_removed(self, content_hash: str) -> None:
        """Forget and announce a fragment that left the node (node hook).

        Args:
            content_hash: Content hash.
        """
        if self.durable:
            self.persistence_writer.forget(content_hash)
        self.event_bus.publish(FragmentRemoved(content_hash))

    def on_membership_change(self, change: str, node_id: str) -> None:
        """Announce a peer joining or leaving.

        Args:
            change: ``"joined"`` or ``"left"``.
            node_id: The peer.
        """
        self.event_bus.publish(PeerJoined(node_id) if change == "joined" else PeerLeft(node_id))

    def refresh_metrics(self) -> None:
        """Update point-in-time gauges; called on every ``/metrics`` scrape."""
        from membrane.transport.metrics import sync_node_metrics

        sync_node_metrics(self.node, self.metrics_node)
        if self.tls is not None:
            expires = cert_not_after(self.tls.server_cert_pem)
            if expires is not None:
                self.metrics_transport.tls_cert_expiry.set(expires.timestamp() - time.time())
        if self.cluster_manager is not None:
            record_replication_lag(self.metrics_registry, self.cluster_manager.membership)
            for peer_id, missing in list(self.cluster_manager.replicator.drift.items()):
                record_merkle_drift(self.metrics_registry, MerkleDrift(missing, "", ""), peer_id)
            for node_id, client in list(self.cluster_manager.membership.clients.items()):
                breaker = getattr(client, "breaker", None)
                if breaker is not None:
                    self.metrics_peers.circuit_open.set(1.0 if breaker.is_open() else 0.0, peer=node_id)
            peers = self.cluster_manager.membership.snapshot()
            self.metrics_cluster.peers_total.set(float(len(peers)))
            self.metrics_cluster.peers_healthy.set(float(sum(1 for p in peers if p.healthy)))

    def restore_fragments(self) -> int:
        """Reload this node's fragments from Redis after a restart.

        A fragment is restored only when it is metadata-only or its KV
        bytes are still in the node's content store (``--data-dir``);
        expired or byte-less entries are dropped from the node's set.

        Returns:
            int: Number of fragments restored.
        """
        if not self.durable:
            return 0
        node_id = self.node.node_id
        restored = 0
        for content_hash in sorted(self.persistence.list_node_fragments(node_id)):
            fragment = self.persistence.retrieve_fragment(content_hash)
            if fragment is None or (
                fragment.payload_ref is not None and not self.node.content_store.has(fragment.payload_ref)
            ):
                self.persistence_writer.forget(content_hash)
                continue
            primary = self.persistence.get_primary(content_hash) == node_id
            if self.node.store(fragment, is_primary=primary):
                restored += 1
        if restored:
            logger.info("Restored %s fragments for %s from Redis", restored, node_id)
        return restored

    def build_transport(self, transport: str, host: str, port: int) -> Any:
        """Build the HTTP transport with authentication, TLS, and quorum wiring.

        Args:
            transport: Transport name; only ``"http"`` is supported.
            host: Bind address.
            port: Listen port.

        Returns:
            Any: The HTTP transport with authentication, TLS, and quorum wiring.
        """
        mtls = self.tls
        if transport != "http":
            raise ValueError(f"unsupported transport={transport!r}; v3.0.0 ships the http transport only")
        server = FastAPIServer(
            node=self.node,
            host=host,
            port=port,
            compute_backend=self.compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=self.cluster_manager,
            metrics_registry=self.metrics_registry,
            tls=mtls,
            authenticator=self.authenticator,
            limits=self.limits,
        )
        # op_store reads these: ``server.is_draining`` rejects writes
        # during drain, and ``quorum_attempt`` makes strong / quorum
        # writes block on replica acks instead of degrading to
        # local-only.
        server.app.state.server = self
        server.app.state.refresh_metrics = self.refresh_metrics
        if self.replicator is not None:
            server.app.state.quorum_attempt = self.replicator
        return server

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Restore durable state, then start the transport and background tasks."""
        # Re-hydrate before any thread starts so the cluster manager,
        # snapshot, and node are populated before live traffic.
        self.persistence_writer.start()
        self.restore_state()
        self.restore_fragments()
        self.running = True
        if self.cluster_manager:
            self.cluster_manager.start()
        self.thread = threading.Thread(target=self.transport.start, daemon=True, name="membrane-http")
        self.thread.start()
        if self.cluster_manager is not None and self.checkpoint_interval_sec > 0:
            self.checkpoint_task.start()
        if self.sweep_interval_sec > 0:
            directory = self.cluster_manager.directory if self.cluster_manager else None
            if directory is not None:
                registry_for_dir: Registry = directory

                def forget_swept(hashes: list[str]) -> None:
                    for content_hash in hashes:
                        registry_for_dir.forget_fragment(content_hash)

                self.sweeper.on_post_sweep = forget_swept
            self.sweeper_task.start()
        if isinstance(getattr(self.node.content_store, "key_provider", None), DirectoryKeyring):
            self.key_refresh_task.start()
        if self.cert_watcher is not None:
            self.cert_watcher.install_sighup()
            self.cert_watcher.start()
        if self.acme is not None:
            self.acme_task.start()
        if self.spiffe is not None:
            self.spiffe_task.start()
        self.log_event("info", f"Server started on {self.host}:{self.port}")

    def rotate_tls(self, cert_pem: str, key_pem: str) -> None:
        """Serve, and present to peers, a renewed certificate and key.

        Args:
            cert_pem: Certificate chain PEM.
            key_pem: Private key PEM.
        """
        if self.tls is None:
            return
        self.tls = dataclasses.replace(
            self.tls,
            server_cert_pem=cert_pem,
            server_key_pem=key_pem,
            client_cert_pem=cert_pem,
            client_key_pem=key_pem,
        )
        if not self.transport.reload_tls(cert_pem, key_pem):
            return
        # Every peer client shares this context, so they all switch at once.
        context = get_default_peer_credentials().ssl_context
        if context is not None and self.transport.tls_tmpdir is not None:
            directory = self.transport.tls_tmpdir.name
            context.load_cert_chain(f"{directory}/server.crt.pem", f"{directory}/server.key.pem")
        self.log_event("info", "TLS certificate rotated")

    def renew_acme_certificate(self) -> bool:
        """Renew the ACME certificate when it nears expiry, and serve the new one.

        Returns:
            bool: True when a new certificate was issued.
        """
        if self.acme is None or not ensure_certificate(self.acme):
            return False
        if self.cert_watcher is not None:
            self.cert_watcher.reload()
        return True

    def refresh_svid(self) -> bool:
        """Fetch the current SVID and serve it when it changed.

        Returns:
            bool: True when a new SVID was installed.
        """
        if self.spiffe is None or self.tls is None:
            return False
        fresh = SPIFFEClient(self.spiffe).fetch_mtls_config()
        if fresh.server_cert_pem == self.tls.server_cert_pem:
            return False
        self.rotate_tls(fresh.server_cert_pem, fresh.server_key_pem)
        return True

    def refresh_data_keys(self) -> int:
        """Activate new data-key versions and re-encrypt blobs under them.

        Returns:
            int: Blobs re-encrypted (0 when no new version appeared or the
            store does not use a key directory).
        """
        store = self.node.content_store
        keyring = getattr(store, "key_provider", None)
        if not isinstance(keyring, DirectoryKeyring) or not keyring.refresh():
            return 0
        rewritten = int(store.reencrypt_all())  # type: ignore[attr-defined]
        logger.info("re-encrypted %s blobs under data key version %s", rewritten, keyring.active_version)
        return rewritten

    def sweep_once(self) -> None:
        """Run one TTL and tombstone sweep (capacity eviction happens on store)."""
        self.sweeper.run_once(evict_expired=self.node.sweep_expired, tombstones=self.tombstones)

    def stop(self, deadline_sec: float = 10.0) -> bool:
        """Stop the server gracefully; later calls are no-ops.

        Order: background tasks, a final checkpoint, the HTTP listener,
        the cluster, outstanding quorum fan-outs, and finally the
        persistence queue, which is flushed within the deadline.

        Args:
            deadline_sec: Budget for each component to stop. Default
                ``10.0`` fits inside a typical SIGTERM grace period.

        Returns:
            bool: True when every component stopped within the budget
            and no persistence write was lost.
        """
        if self.__stopped:
            return True
        self.__stopped = True
        joined_cleanly = self.checkpoint_task.stop(deadline_sec)
        # A final checkpoint lets the next process rebuild from
        # up-to-date state.
        self.checkpoint_state()
        self.running = False
        self.transport.stop()
        if self.cluster_manager:
            self.cluster_manager.stop(deadline_sec=deadline_sec)
        joined_cleanly = self.sweeper_task.stop(deadline_sec) and joined_cleanly
        self.key_refresh_task.stop(deadline_sec)
        if self.cert_watcher is not None:
            self.cert_watcher.stop()
        self.acme_task.stop(deadline_sec)
        self.spiffe_task.stop(deadline_sec)
        if self.replicator is not None:
            self.replicator.shutdown()
        joined_cleanly = self.persistence_writer.stop(deadline_sec) and joined_cleanly
        self.event_bus.close(timeout_sec=min(deadline_sec, 5.0))
        TRACING.shutdown()  # flush spans still queued for export
        self.log_event("info", f"Server stopped (cleanly={joined_cleanly})")
        return joined_cleanly

    def drain(self, deadline_sec: float = 30.0) -> dict[str, Any]:
        """Best-effort drain: stop accepting writes, migrate primaries, leave cluster.

        1. Mark ``self.is_draining = True``. ``/readyz`` and
           :func:`op_store` then return 503 with ``Retry-After``.
        2. Iterate every hash where this node is the primary and
           hand it, bytes included, to its next healthy ring owner
           (:meth:`Replicator.hand_off`, verified before ownership moves).
        3. Sleep at most ``deadline_sec`` for the migration pass
           to finish (or for ``is_draining`` to be reset by a
           concurrent operator). On timeout log the stragglers.
        4. Call :meth:`Membership.leave_cluster` so peers converge
           without a competing lease window, then :meth:`stop`.

        Args:
            deadline_sec: Wall-clock budget for the drain.

        Returns:
            dict[str, int]: ``{"migrated": ..., "stragglers": ...,
            "duration_sec": ...}`` for operator logs / metrics.
        """
        self.is_draining = True
        self.event_bus.publish(DrainStarted(deadline_sec))
        self.log_event("info", f"Drain started with deadline={deadline_sec}s")
        start = time.time()

        # Step 2: hand every local primary, bytes included, to its next
        # healthy ring owner, verified before this node lets go of it.
        migrated = 0
        stragglers: list[str] = []
        if self.cluster_manager is not None:
            replicator = self.cluster_manager.replicator
            ring = self.cluster_manager.shard_manager.hash_ring
            healthy = {peer.node_id for peer in self.cluster_manager.membership.healthy()}
            healthy.discard(self.node.node_id)
            for content_hash in sorted(self.node.get_shard_hashes()):
                if time.time() - start >= deadline_sec:
                    stragglers.append(content_hash)
                    continue
                target = self.drain_target(content_hash, ring, healthy)
                if target is not None and replicator.hand_off(content_hash, target):
                    migrated += 1
                else:
                    stragglers.append(content_hash)
            if stragglers:
                logger.warning(
                    "drain: %s primaries could not be handed off; their replicas still serve them",
                    len(stragglers),
                )

        # Step 3: best-effort deadline while the leave propagates.
        elapsed = time.time() - start
        remaining = deadline_sec - elapsed
        if remaining > 0 and not stragglers:
            time.sleep(min(remaining, 1.0))

        duration = time.time() - start
        self.log_event(
            "info",
            f"Drain finished in {duration:.1f}s; migrated={migrated} stragglers={len(stragglers)}",
        )

        # Step 4: graceful leave + stop. Single-node deployments
        # have no cluster_manager so the leave is a no-op.
        if self.cluster_manager is not None:
            try:
                self.cluster_manager.membership.leave_cluster()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("drain: leave_cluster raised: %s", exc)

        self.event_bus.publish(DrainFinished(migrated, len(stragglers)))
        self.stop()
        return {
            "migrated": migrated,
            "stragglers": len(stragglers),
            "duration_sec": duration,
        }

    def drain_target(self, content_hash: str, ring: Any, healthy: set[str]) -> str | None:
        """Pick the node that takes over a primary during drain.

        Args:
            content_hash: The primary being handed off.
            ring: The cluster's hash ring.
            healthy: Healthy peer ids, excluding this node.

        Returns:
            str | None: The first healthy node in the hash's ring order, any
            healthy peer when the ring has none, or ``None`` when no peer is
            healthy.
        """
        if not healthy:
            return None
        try:
            ordered = ring.get_nodes(content_hash, n=len(healthy) + 1)
        except Exception:
            ordered = []
        for node_id in ordered:
            if node_id in healthy:
                return str(node_id)
        return sorted(healthy)[0]

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

    def join(self, deadline_sec: float | None = None) -> bool:
        """Block until the server thread exits.

        Args:
            deadline_sec: Optional wall-clock budget. When set,
                :meth:`threading.Thread.join` is called with the
                deadline; the caller can decide whether to
                escalate. ``None`` (the default) blocks
                indefinitely, matching the behaviour of the
                underlying :meth:`threading.Thread.join`.

        Returns:
            bool: ``True`` when the thread exited within the
            budget (or ``None`` was supplied and the thread has
            exited); ``False`` when a deadline was supplied and
            the thread is still alive at the end of the budget.
        """
        if self.thread is None:
            return True
        self.thread.join(timeout=deadline_sec)
        return not self.thread.is_alive()

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
        """Record a server event for the dashboard (the newest 10,000 are kept).

        Args:
            level: Event level (``info``, ``warn``, ``error``).
            message: Human-readable description.
            node_id: Node identifier.
            bytes_affected: Optional size in bytes associated with the event.
        """
        self.events.record(level, message, node_id, bytes_affected)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> ServerDiagnostics:
        """Return a current snapshot of server health.

        Returns:
            ServerDiagnostics: A current snapshot of server health.
        """
        stats = self.node.get_stats()
        now = time.time()
        hits = int(self.metrics_node.cache_lookups.get(result="hit"))
        misses = int(self.metrics_node.cache_lookups.get(result="miss"))
        lookups = CacheMetrics(hits=hits, misses=misses, total_requests=hits + misses)
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
            hit_rate=lookups.hit_rate(),
            miss_rate=lookups.miss_rate(),
            request_count=self.request_count,
            error_count=self.error_count,
            connected_nodes=connected,
            backend_name=self.compute_backend.device_name(),
            redis_connected=self.persistence.ping(),
            load=self.node.heartbeat(),
        )

    def recent_events(self, n: int = 20) -> list[ServerEvent]:
        """Return the last ``n`` events.

        Args:
            n: Number of events to return.

        Returns:
            list[ServerEvent]: The last ``n`` events.
        """
        return self.events.recent(n)


__all__ = [
    "Server",
    "ServerDiagnostics",
    "ServerEvent",
    "build_content_store",
    "resolves_to_loopback",
]
