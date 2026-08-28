"""FastAPIServer: production HTTP/REST server using FastAPI + uvicorn.

Mirrors the stdlib HTTP transport in :mod:`membrane.transport.http`
with identical request/response shapes but native async handlers
and Pydantic request bodies. The shared business logic lives in
:mod:`membrane.transport.routes`; this module is purely the
async transport binding.

Endpoints (identical contract to the stdlib transport):

* ``POST /store`` — store a fragment.
* ``GET /retrieve`` — retrieve a fragment by ``content_hash``.
* ``GET /inventory`` — return the node's inventory digest.
* ``POST /sync`` — sync missing fragments from a source URL.
* ``GET /heartbeat`` — node health and load snapshot.
* ``POST /prefill`` — run prefill and return fragments.
* ``POST /join`` — join the cluster.
* ``POST /leave`` — leave the cluster.
* ``POST /gossip`` — exchange gossip state.
* ``GET /peers`` — list known peers.
* ``POST /replicate`` — store a fragment as a replica.
* ``GET /metrics`` — Prometheus text exposition.
* ``GET /metrics.json`` — legacy JSON for the TUI.
* ``GET /livez`` — process liveness probe.
* ``GET /readyz`` — deep readiness probe.

Observability:
    * ``/livez`` — process liveness probe.
    * ``/readyz`` — deep readiness probe.
    * ``/metrics`` — Prometheus text exposition.
    * ``/metrics.json`` — legacy JSON snapshot for the TUI.

Security:
    * The server is unauthenticated by default. Place it behind an
      authenticating reverse proxy in production, or wire an
      :class:`Authenticator` into the request pipeline (see
      :mod:`membrane.auth`).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI

from membrane.compute.base import Backend
from membrane.metrics import MetricsCollector
from membrane.node import Node
from membrane.transfer import TransferService
from membrane.transport.routes_fastapi import register_routes

logger = logging.getLogger(__name__)


def create_app(
    node: Node,
    compute_backend: Backend | None,
    transfer_service: TransferService,
    cluster_manager: Any | None,
    metrics_registry: MetricsCollector | None = None,
) -> FastAPI:
    """Build a configured FastAPI application for a Membrane node.

    Args:
        node: Local :class:`Node`.
        compute_backend: Optional :class:`Backend`.
        transfer_service: :class:`TransferService`.
        cluster_manager: Optional cluster manager.
        metrics_registry: Optional :class:`MetricsCollector` for the
            ``/metrics`` Prometheus endpoint. When ``None``, ``/metrics``
            falls back to a JSON snapshot of the node's stats.

    Returns:
        FastAPI: Configured application ready to be served by
        uvicorn.
    """
    app = FastAPI(title="Membrane", version="0.1.0")
    app.state.node = node
    app.state.compute_backend = compute_backend
    app.state.transfer_service = transfer_service
    app.state.cluster_manager = cluster_manager
    app.state.metrics_registry = metrics_registry

    register_routes(app)
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
    """

    def __init__(
        self,
        node: Node,
        host: str = "0.0.0.0",
        port: int = 8080,
        compute_backend: Backend | None = None,
        transfer_service: TransferService | None = None,
        cluster_manager: Any | None = None,
        metrics_registry: MetricsCollector | None = None,
    ) -> None:
        """Initialize the FastAPI server wrapper."""
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service or TransferService()
        self.cluster_manager = cluster_manager
        self.metrics_registry = metrics_registry
        self.app = create_app(
            node=node,
            compute_backend=compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=cluster_manager,
            metrics_registry=metrics_registry,
        )

    def start(self) -> None:
        """Start uvicorn serving the configured app.

        Blocks until :meth:`stop` is called.
        """
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        logger.info("FastAPI server listening on http://%s:%s", self.host, self.port)
        self._server.run()

    def stop(self) -> None:
        """Stop the uvicorn server.

        Sets ``should_exit = True`` on the underlying server; the
        blocking ``run()`` returns shortly thereafter.
        """
        if self._server:
            self._server.should_exit = True
            logger.info("FastAPI server stopped")

    def run_in_thread(self) -> None:
        """Start the server in a background daemon thread."""
        import threading

        t = threading.Thread(target=self.start, daemon=True)
        t.start()
