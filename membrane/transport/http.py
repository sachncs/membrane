"""HTTPServer: production HTTP/REST server for Membrane nodes.

Uses the Python standard library :mod:`http.server` so the HTTP
transport has zero external dependencies. All request and response
bodies are JSON.

Endpoints:

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
* ``GET /metrics`` — extended node metrics.
* ``GET /livez`` — process liveness probe.
* ``GET /readyz`` — deep readiness probe.

Implementation:

* :class:`HTTPServer` — public façade used by callers.
* :class:`StdlibServer` — ``http.server.HTTPServer`` subclass that
  holds references to the local node, compute backend, transfer
  service, and cluster manager.
* :class:`Handler` — request handler that dispatches via the
  ``ROUTES`` table to module-level handler functions in
  :mod:`membrane.transport.routes`. The class itself carries only
  ``do_GET``, ``do_POST``, ``log_message``, ``send_json``, and
  ``read_json``.

Security:
    * The HTTP server is unauthenticated. Restrict exposure
      with a reverse proxy or run inside a trusted network
      boundary.
    * No request body is rate-limited; add a reverse proxy in
      front of the listener if abuse is a concern.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer as StdlibHTTPServer
from typing import Any

from membrane.compute.base import Backend
from membrane.node import Node
from membrane.transport.routes import ROUTES, MAX_BODY_BYTES
from membrane.transfer import TransferService

logger = logging.getLogger(__name__)


class StdlibServer(StdlibHTTPServer):
    """Custom HTTPServer that holds references to node, compute, and cluster.

    Attributes:
        node: Local :class:`Node`.
        compute_backend: Optional :class:`Backend` used by
            ``POST /prefill``.
        transfer_service: Transfer service used by ``POST /sync``.
        cluster_manager: Optional cluster manager used by the
            membership endpoints.
    """

    def __init__(
        self,
        server_address,
        handler_class,
        node: Node,
        compute_backend: Backend | None,
        transfer_service: TransferService,
        cluster_manager: Any | None,
    ) -> None:
        """Initialize the underlying HTTP server with extra state."""
        super().__init__(server_address, handler_class)
        self.node = node
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service
        self.cluster_manager = cluster_manager


class Handler(BaseHTTPRequestHandler):
    """Request handler for Membrane HTTP transport.

    The handler is a thin dispatcher: it parses the URL path,
    reads any JSON body, and forwards to a module-level handler
    function from :data:`membrane.transport.routes.ROUTES`.
    JSON serialization is delegated to :meth:`send_json` /
    :meth:`read_json`.
    """

    server: StdlibServer  # type: ignore[misc]

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route stdlib HTTP server logs through the Membrane logger."""
        logger.debug(fmt, *args)

    def send_json(self, status: int, data: dict[str, Any]) -> None:
        """Send a JSON response with the given status code.

        Args:
            status: HTTP status code.
            data: JSON-serializable payload.
        """
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def read_json(self) -> dict[str, Any]:
        """Read the request body and parse it as JSON.

        Returns:
            dict[str, Any]: Parsed payload. Empty dict when no
            body is supplied.

        Raises:
            ValueError: If Content-Length exceeds MAX_BODY_BYTES.
        """
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_BYTES:
            raise ValueError(f"body too large: {length} > {MAX_BODY_BYTES}")
        body = self.rfile.read(length)
        return json.loads(body.decode()) if body else {}

    def do_GET(self) -> None:
        """Dispatch GET requests via the route table."""
        path = self.path.split("?")[0]
        handler = ROUTES.get(("GET", path))
        if handler is None:
            self.send_json(404, {"error": "not found"})
            return
        try:
            handler(self)
        except Exception as exc:
            logger.exception("GET %s failed", path)
            self.send_json(500, {"error": "internal"})

    def do_POST(self) -> None:
        """Dispatch POST requests via the route table."""
        path = self.path.split("?")[0]
        handler = ROUTES.get(("POST", path))
        if handler is None:
            self.send_json(404, {"error": "not found"})
            return
        try:
            handler(self)
        except Exception as exc:
            logger.exception("POST %s failed", path)
            self.send_json(500, {"error": "internal"})


class HTTPServer:
    """Production HTTP server wrapping a Node.

    Args:
        node: Node to serve.
        host: Bind address.
        port: Listen port.
        compute_backend: Optional compute backend for prefill.
        transfer_service: Optional transfer service for sync.
        cluster_manager: Optional cluster manager for peer
            management.
    """

    def __init__(
        self,
        node: Node,
        host: str = "0.0.0.0",
        port: int = 8080,
        compute_backend: Backend | None = None,
        transfer_service: TransferService | None = None,
        cluster_manager: Any | None = None,
    ) -> None:
        """Initialize the HTTP server wrapper."""
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service or TransferService()
        self.cluster_manager = cluster_manager
        self.server: StdlibServer | None = None

    def start(self) -> None:
        """Start the HTTP server (blocking).

        Calls :meth:`http.server.HTTPServer.serve_forever` on the
        underlying server. The caller is expected to invoke
        :meth:`stop` from another thread to terminate the serve
        loop.
        """
        self.server = StdlibServer(
            (self.host, self.port),
            Handler,
            node=self.node,
            compute_backend=self.compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=self.cluster_manager,
        )
        logger.info("HTTP server listening on http://%s:%s", self.host, self.port)
        self.server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server.

        No-op when the server was never started.
        """
        if self.server:
            self.server.shutdown()
            logger.info("HTTP server stopped")

    def run_in_thread(self) -> None:
        """Start the server in a background daemon thread."""
        import threading

        t = threading.Thread(target=self.start, daemon=True)
        t.start()
