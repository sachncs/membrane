"""FastAPIServer: production HTTP/REST server using FastAPI + uvicorn.

The shared business logic lives in
:mod:`membrane.transport.ops`; this module is purely the
async transport binding.

Endpoints:

* ``POST /store`` -- store a fragment.
* ``GET /retrieve`` -- retrieve a fragment by ``content_hash``.
* ``GET /inventory`` -- return the node's inventory digest.
* ``POST /sync`` -- sync missing fragments from a source URL.
* ``GET /heartbeat`` -- node health and load snapshot.
* ``POST /prefill`` -- run prefill and return fragments.
* ``POST /join`` -- join the cluster.
* ``POST /leave`` -- leave the cluster.
* ``POST /gossip`` -- exchange gossip state.
* ``GET /peers`` -- list known peers.
* ``POST /replicate`` -- store a fragment as a replica.
* ``GET /metrics`` -- Prometheus text exposition.
* ``GET /metrics.json`` -- legacy JSON for the TUI.
* ``GET /livez`` -- process liveness probe.
* ``GET /readyz`` -- readiness probe (503 while draining).
* ``GET /openapi.json`` -- API schema; only with
  ``TransportLimits.enable_api_docs``, behind the ``read`` scope.

Capacity: :mod:`membrane.transport.limits` bounds in-flight requests
(503 + ``Retry-After`` when saturated) and optionally rate-limits each
credential (429).

Observability:
    * ``/livez`` -- process liveness probe.
    * ``/readyz`` -- deep readiness probe.
    * ``/metrics`` -- Prometheus text exposition.

Security:
    * When an :class:`~membrane.auth.Authenticator` is attached
      (API keys or mTLS), every endpoint except ``/livez`` and
      ``/readyz`` authenticates the caller and enforces the scope
      from :data:`membrane.transport.authz.ROUTE_SCOPES`. Failures
      return ``401`` (unauthenticated) or ``403`` (missing scope).
    * Multi-node deployments should attach
      :class:`membrane.transport.tls.MTLSConfig`; the peer CN used
      for authorization comes from the verified certificate, never
      from a request header.
"""

import logging
import os
from typing import Any

from fastapi import FastAPI

from membrane.auth import Authenticator
from membrane.compute.base import Backend
from membrane.metrics import MetricsCollector
from membrane.network.cluster import Cluster
from membrane.node import Node
from membrane.transfer import TransferService
from membrane.transport.limits import ConcurrencyLimitMiddleware, RateLimitMiddleware, TransportLimits
from membrane.transport.request_id import RequestIdMiddleware
from membrane.transport.routes import register_routes
from membrane.transport.tls import MTLSConfig, build_server_context
from membrane.transport.tls_protocol import PeerCertH11Protocol
from membrane.transport.tracing import TracingMiddleware

logger = logging.getLogger(__name__)


def create_app(
    node: Node,
    compute_backend: Backend | None,
    transfer_service: TransferService,
    cluster_manager: Cluster | None,
    metrics_registry: MetricsCollector | None = None,
    authenticator: Authenticator | None = None,
    limits: TransportLimits | None = None,
) -> FastAPI:
    """Build a configured FastAPI application for a Membrane node.

    Args:
        node: Local :class:`Node`.
        compute_backend: Optional :class:`Backend`.
        transfer_service: :class:`TransferService`.
        cluster_manager: Optional :class:`Cluster`.
        metrics_registry: Optional :class:`MetricsCollector` for the
            ``/metrics`` Prometheus endpoint. When ``None``, ``/metrics``
            falls back to a JSON snapshot of the node's stats.
        authenticator: Optional :class:`~membrane.auth.Authenticator`.
            When set, every route except ``/livez`` and ``/readyz``
            authenticates the caller and enforces its route scope.
        limits: Concurrency, rate-limit, and API-docs settings; defaults
            to :class:`TransportLimits`.

    Returns:
        FastAPI: Configured application ready to be served by
        uvicorn.
    """
    from membrane import __version__

    limits = limits or TransportLimits()
    # FastAPI's built-in /docs, /redoc and /openapi.json bypass route
    # authentication, so they stay off; register_routes serves the schema
    # behind the read scope when enable_api_docs is set.
    app = FastAPI(title="Membrane", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.limits = limits
    app.state.node = node
    app.state.compute_backend = compute_backend
    app.state.transfer_service = transfer_service
    app.state.cluster_manager = cluster_manager
    app.state.metrics_registry = metrics_registry
    app.state.authenticator = authenticator
    if metrics_registry is not None:
        from membrane.metrics import ClusterMetrics, TransportMetrics

        app.state.transport_metrics = TransportMetrics(metrics_registry)
        app.state.cluster_metrics = ClusterMetrics(metrics_registry)
    else:
        app.state.transport_metrics = None
        app.state.cluster_metrics = None

    register_routes(app)
    on_reject = None
    if app.state.transport_metrics is not None:
        rejected = app.state.transport_metrics.rejected

        def on_reject(reason: str) -> None:
            rejected.inc(reason=reason)

    # Starlette runs the last-added middleware first: request IDs wrap
    # everything (so 429 / 503 responses carry one), then rate limiting,
    # then the concurrency bound.
    if limits.max_concurrency > 0:
        app.add_middleware(
            ConcurrencyLimitMiddleware,
            max_concurrency=limits.max_concurrency,
            queue_timeout_sec=limits.queue_timeout_sec,
            on_reject=on_reject,
        )
    if limits.rate_limit_per_sec > 0:
        app.add_middleware(
            RateLimitMiddleware,
            rate_per_sec=limits.rate_limit_per_sec,
            burst=limits.rate_limit_burst or max(1, round(2 * limits.rate_limit_per_sec)),
            on_reject=on_reject,
        )
    # Tracing runs inside the request-ID middleware so spans carry the ID.
    app.add_middleware(TracingMiddleware)
    app.add_middleware(RequestIdMiddleware)
    try:
        from membrane.transport.admin import create_admin_router

        app.include_router(create_admin_router(), prefix="")
    except ImportError:  # pragma: no cover - the admin router is optional
        pass
    return app


class FastAPIServer:
    """Production HTTP server using FastAPI + uvicorn.

    Args:
        node: Node to serve.
        host: Bind address.
        port: Listen port.
        compute_backend: Optional compute backend for prefill.
        transfer_service: Optional transfer service for sync.
        cluster_manager: Optional cluster manager for peer
            management.
        metrics_registry: Optional :class:`MetricsCollector` for the
            ``/metrics`` Prometheus endpoint.
        tls: Optional :class:`MTLSConfig`. When supplied,
            :meth:`start` builds a server-side SSLContext via
            :func:`membrane.transport.tls.build_server_context`
            and feeds it to ``uvicorn.Config(ssl_context=...)``.
            uvicorn terminates the handshake and writes the
            uvicorn terminates the handshake and
            :class:`~membrane.transport.tls_protocol.PeerCertH11Protocol`
            records the verified peer cert's CN in the request
            scope, where :class:`MTLSAuthenticator` reads it.
        authenticator: Optional authenticator enforced on every
            non-probe route.
        limits: Capacity settings (see :class:`TransportLimits`).
    """

    def __init__(
        self,
        node: Node,
        host: str = "0.0.0.0",
        port: int = 8080,
        compute_backend: Backend | None = None,
        transfer_service: TransferService | None = None,
        cluster_manager: Cluster | None = None,
        metrics_registry: MetricsCollector | None = None,
        tls: MTLSConfig | None = None,
        authenticator: Authenticator | None = None,
        limits: TransportLimits | None = None,
    ) -> None:
        """Initialize the FastAPI server wrapper.

        Args:
            node: Local :class:`Node`.
            host: Bind address.
            port: Listen port.
            compute_backend: Optional :class:`Backend`.
            transfer_service: :class:`TransferService`.
            cluster_manager: Optional :class:`Cluster`.
            metrics_registry: Optional :class:`MetricsCollector` for the
                ``/metrics`` Prometheus endpoint. When ``None``, ``/metrics``
                falls back to a JSON snapshot of the node's stats.
            tls: mTLS configuration; defaults to ``cluster_config.mtls``.
            authenticator: Optional :class:`~membrane.auth.Authenticator`. When
                set, every route except ``/livez`` and ``/readyz`` authenticates
                the caller and enforces its route scope.
            limits: Concurrency, rate-limit, connection, and API-docs settings.
        """
        self.node = node
        self.limits = limits or TransportLimits()
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service or TransferService()
        self.cluster_manager = cluster_manager
        self.metrics_registry = metrics_registry
        self.tls = tls
        self.server: Any | None = None
        self.tls_tmpdir: Any | None = None
        self.app = create_app(
            node=node,
            compute_backend=compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=cluster_manager,
            metrics_registry=metrics_registry,
            authenticator=authenticator,
            limits=self.limits,
        )

    def start(self) -> None:
        """Start uvicorn serving the configured app.

        Blocks until :meth:`stop` is called.
        """
        import tempfile

        import uvicorn

        ssl_kwargs: dict[str, Any] = {}
        if self.tls is not None:
            # Build a real SSLContext first to validate the chain
            # eagerly — uvicorn surfaces later, opaque errors.
            build_server_context(self.tls)
            # uvicorn.Config takes file paths for the cert chain
            # and CA bundle. We write the configured PEMs to a
            # short-lived tmpdir; cleanup happens in ``stop`` so
            # the lifetime matches the running server.
            self.tls_tmpdir = tempfile.TemporaryDirectory(prefix="membrane-tls-")
            cert_path = f"{self.tls_tmpdir.name}/server.crt.pem"
            key_path = f"{self.tls_tmpdir.name}/server.key.pem"
            ca_path = f"{self.tls_tmpdir.name}/ca-bundle.pem"
            with open(cert_path, "w") as f:
                f.write(self.tls.server_cert_pem)
            with open(key_path, "w") as f:
                f.write(self.tls.server_key_pem)
            with open(ca_path, "w") as f:
                f.write(self.tls.ca_bundle_pem)
            ssl_kwargs = {
                "ssl_certfile": cert_path,
                "ssl_keyfile": key_path,
                "ssl_ca_certs": ca_path if self.tls.ca_bundle_pem else None,
                "ssl_cert_reqs": 2 if self.tls.require_client_cert else 0,
                # Surfaces the verified peer cert CN to the authenticator.
                "http": PeerCertH11Protocol,
            }
            logger.info(
                "FastAPI mTLS enabled: require_client_cert=%s",
                self.tls.require_client_cert,
            )
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            # Keep uvicorn's records on Membrane's handlers (text or JSON).
            log_config=None,
            access_log=False,
            limit_concurrency=self.limits.max_connections,
            timeout_keep_alive=int(self.limits.keep_alive_timeout_sec),
            timeout_graceful_shutdown=10,
            **ssl_kwargs,
        )
        self.server = uvicorn.Server(config)
        scheme = "https" if ssl_kwargs else "http"
        logger.info("FastAPI server listening on %s://%s:%s", scheme, self.host, self.port)
        try:
            self.server.run()
        finally:
            tmp = getattr(self, "tls_tmpdir", None)
            if tmp is not None:
                tmp.cleanup()
                self.tls_tmpdir = None

    def reload_tls(self, cert_pem: str, key_pem: str) -> bool:
        """Serve a new certificate and key on new connections, without a restart.

        Args:
            cert_pem: Certificate chain PEM.
            key_pem: Private key PEM.

        Returns:
            bool: True when the live listener now uses the new chain; False
            when TLS is off or the server has not started.
        """
        config = getattr(self.server, "config", None)
        context = getattr(config, "ssl", None)
        if self.tls_tmpdir is None or context is None:
            return False
        cert_path = f"{self.tls_tmpdir.name}/server.crt.pem"
        key_path = f"{self.tls_tmpdir.name}/server.key.pem"
        with open(cert_path, "w") as f:
            f.write(cert_pem)
        fd = os.open(key_path, os.O_WRONLY | os.O_TRUNC | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key_pem)
        context.load_cert_chain(cert_path, key_path)
        logger.info("listener certificate reloaded")
        return True

    def stop(self) -> None:
        """Stop the uvicorn server.

        Sets ``should_exit = True`` on the underlying server; the
        blocking ``run()`` returns shortly thereafter.
        """
        if self.server:
            self.server.should_exit = True
            logger.info("FastAPI server stopped")

    def run_in_thread(self) -> None:
        """Start the server in a background daemon thread."""
        import threading

        t = threading.Thread(target=self.start, daemon=True)
        t.start()


__all__ = [
    "FastAPIServer",
    "create_app",
]
