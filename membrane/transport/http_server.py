"""HTTPServer: production HTTP/REST server for Membrane nodes.

Uses the Python standard library :mod:`http.server` so the HTTP
transport has zero external dependencies. All request and
response bodies are JSON.

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

The implementation has three nested classes:

* :class:`HTTPServer` — public façade used by callers.
* :class:`MembraneStdlibHTTPServer` — ``http.server.HTTPServer``
  subclass that holds references to the local node, compute
  backend, transfer service, and cluster manager.
* :class:`MembraneHTTPHandler` — request handler that
  dispatches to one of the ``_handle_*`` methods.

Security:
    * The HTTP server is unauthenticated. Restrict exposure
      with a reverse proxy or run inside a trusted network
      boundary.
    * No request body is rate-limited; add a reverse proxy in
      front of the listener if abuse is a concern.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer as StdlibHTTPServer
from typing import Any

from membrane.compute.backend import ComputeBackend
from membrane.compute.cpu_backend import CPUBackend
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature
from membrane.transfer_service import TransferService

logger = logging.getLogger(__name__)


def serialize_fragment(frag: Fragment) -> dict[str, Any]:
    """Serialize a fragment to a JSON-compatible dict.

    Args:
        frag: Fragment to serialize.

    Returns:
        dict[str, Any]: Flat dict suitable for HTTP transport.
    """
    return {
        "content_hash": frag.content_hash,
        "embedding": list(frag.embedding),
        "model_id": frag.structural_signature.model_id,
        "layer_range": frag.structural_signature.layer_range,
        "token_span": frag.structural_signature.token_span,
        "size": frag.size,
        "ttl": frag.ttl,
        "reuse_score": frag.reuse_score,
        "version_id": frag.version_id,
    }


def deserialize_fragment(data: dict[str, Any]) -> Fragment:
    """Reconstruct a fragment from its serialized form.

    Args:
        data: Mapping produced by :func:`serialize_fragment`.

    Returns:
        Fragment: Reconstructed fragment instance.
    """
    return Fragment(
        content_hash=data["content_hash"],
        embedding=tuple(data["embedding"]),
        structural_signature=StructuralSignature(
            model_id=data["model_id"],
            layer_range=tuple(data["layer_range"]),
            token_span=tuple(data["token_span"]),
        ),
        size=data["size"],
        ttl=data["ttl"],
        reuse_score=data["reuse_score"],
        version_id=data["version_id"],
    )


class MembraneStdlibHTTPServer(StdlibHTTPServer):
    """Custom HTTPServer that holds references to node, compute, and cluster.

    Attributes:
        node: Local :class:`MembraneNode`.
        compute_backend: Optional :class:`ComputeBackend` used by
            ``POST /prefill``.
        transfer_service: Transfer service used by
            ``POST /sync``.
        cluster_manager: Optional cluster manager used by the
            membership endpoints.
    """

    def __init__(
        self,
        server_address,
        handler_class,
        node: MembraneNode,
        compute_backend: ComputeBackend | None,
        transfer_service: TransferService,
        cluster_manager: Any | None,
    ) -> None:
        """Initialize the underlying HTTP server with extra state."""
        super().__init__(server_address, handler_class)
        self.node = node
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service
        self.cluster_manager = cluster_manager


class MembraneHTTPHandler(BaseHTTPRequestHandler):
    """Request handler for Membrane HTTP transport.

    The handler is a thin dispatcher: it parses the URL path,
    reads any JSON body, and forwards to one of the
    ``_handle_*`` methods. JSON serialization is delegated to
    :meth:`send_json` / :meth:`read_json`.
    """

    server: MembraneStdlibHTTPServer  # type: ignore[misc]

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
        """
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body.decode()) if body else {}

    def do_GET(self) -> None:
        """Dispatch GET requests to the appropriate handler."""
        path = self.path.split("?")[0]
        if path == "/retrieve":
            self.handle_retrieve()
        elif path == "/inventory":
            self.handle_inventory()
        elif path == "/heartbeat":
            self.handle_heartbeat()
        elif path == "/metrics":
            self.handle_metrics()
        elif path == "/peers":
            self.handle_peers()
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        """Dispatch POST requests to the appropriate handler."""
        path = self.path.split("?")[0]
        if path == "/store":
            self.handle_store()
        elif path == "/sync":
            self.handle_sync()
        elif path == "/prefill":
            self.handle_prefill()
        elif path == "/join":
            self.handle_join()
        elif path == "/leave":
            self.handle_leave()
        elif path == "/gossip":
            self.handle_gossip()
        elif path == "/replicate":
            self.handle_replicate()
        else:
            self.send_json(404, {"error": "not found"})

    def handle_store(self) -> None:
        """Handle ``POST /store``."""
        data = self.read_json()
        try:
            frag = deserialize_fragment(data["fragment"])
            is_primary = data.get("is_primary", False)
            ok = self.server.node.store(frag, is_primary=is_primary) if self.server.node else False
            self.send_json(200 if ok else 500, {"success": ok, "content_hash": frag.content_hash})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_replicate(self) -> None:
        """Handle ``POST /replicate`` — store a fragment as a replica."""
        data = self.read_json()
        try:
            frag = deserialize_fragment(data["fragment"])
            ok = self.server.node.store(frag, is_primary=False) if self.server.node else False
            self.send_json(200 if ok else 500, {"success": ok, "content_hash": frag.content_hash})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_retrieve(self) -> None:
        """Handle ``GET /retrieve?content_hash=...``."""
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(self.path).query)
        h = qs.get("content_hash", [None])[0]
        if not h:
            self.send_json(400, {"error": "missing content_hash"})
            return
        frag = self.server.node.retrieve(h) if self.server.node else None
        if frag:
            self.send_json(200, {"found": True, "fragment": serialize_fragment(frag)})
        else:
            self.send_json(404, {"found": False, "fragment": None})

    def handle_inventory(self) -> None:
        """Handle ``GET /inventory``."""
        digest = {h: frag.version_id for h, frag in (self.server.node.fragments.items() if self.server.node else {})}
        self.send_json(200, {"node_id": self.server.node.node_id if self.server.node else "", "digest": digest})

    def handle_heartbeat(self) -> None:
        """Handle ``GET /heartbeat``."""
        if not self.server.node:
            self.send_json(500, {"error": "no node"})
            return
        stats = self.server.node.get_stats()
        self.send_json(200, {
            "node_id": self.server.node.node_id,
            "load": self.server.node.heartbeat(),
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "healthy": True,
        })

    def handle_metrics(self) -> None:
        """Handle ``GET /metrics`` — extended metrics payload."""
        if not self.server.node:
            self.send_json(500, {"error": "no node"})
            return
        stats = self.server.node.get_stats()
        self.send_json(200, {
            "node_id": self.server.node.node_id,
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "load": self.server.node.heartbeat(),
        })

    def handle_sync(self) -> None:
        """Handle ``POST /sync`` — pull missing fragments from a source URL.

        Reads ``source_url`` from the body, fetches the remote
        inventory, computes the missing set against the local
        inventory, and pulls each missing fragment via
        ``GET /retrieve``. Each successfully pulled fragment is
        stored locally as a non-primary replica.
        """
        data = self.read_json()
        source_url = data.get("source_url", "")
        if not source_url:
            self.send_json(400, {"error": "missing source_url"})
            return
        try:
            # Pull inventory from remote and transfer missing fragments.
            import urllib.request
            req = urllib.request.Request(f"{source_url}/inventory")
            with urllib.request.urlopen(req, timeout=5) as resp:
                remote_data = json.loads(resp.read().decode())
            remote_digest = remote_data.get("digest", {})
            local_digest = self.server.transfer_service.inventory_digest(self.server.node)
            missing = self.server.transfer_service.compare_inventories(local_digest, remote_digest)
            transferred: list[str] = []
            for h in missing:
                req = urllib.request.Request(f"{source_url}/retrieve?content_hash={h}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    remote_frag_data = json.loads(resp.read().decode())
                if remote_frag_data.get("found"):
                    frag = deserialize_fragment(remote_frag_data["fragment"])
                    if self.server.node.store(frag, is_primary=False):
                        transferred.append(h)
            self.send_json(200, {"success": True, "transferred": transferred})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_prefill(self) -> None:
        """Handle ``POST /prefill`` — run prefill and store fragments as primary."""
        data = self.read_json()
        tokens = data.get("prompt_tokens", [])
        model_id = data.get("model_id", "default")
        backend = self.server.compute_backend or CPUBackend()
        try:
            fragments = backend.prefill(tokens, model_id)
            for frag in fragments:
                if self.server.node:
                    self.server.node.store(frag, is_primary=True)
            self.send_json(200, {
                "success": True,
                "fragments": [serialize_fragment(f) for f in fragments],
            })
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_join(self) -> None:
        """Handle ``POST /join``."""
        data = self.read_json()
        node_id = data.get("node_id", "")
        host = data.get("host", "")
        port = data.get("port", 0)
        if not node_id or not host or not port:
            self.send_json(400, {"error": "missing node_id, host, or port"})
            return
        if self.server.cluster_manager:
            result = self.server.cluster_manager.on_peer_join(node_id, host, port)
            self.send_json(200, result)
        else:
            self.send_json(503, {"error": "cluster manager not enabled"})

    def handle_leave(self) -> None:
        """Handle ``POST /leave``."""
        data = self.read_json()
        node_id = data.get("node_id", "")
        if not node_id:
            self.send_json(400, {"error": "missing node_id"})
            return
        if self.server.cluster_manager:
            self.server.cluster_manager.on_peer_leave(node_id)
            self.send_json(200, {"success": True})
        else:
            self.send_json(503, {"error": "cluster manager not enabled"})

    def handle_gossip(self) -> None:
        """Handle ``POST /gossip``."""
        data = self.read_json()
        if self.server.cluster_manager:
            result = self.server.cluster_manager.on_gossip(data)
            self.send_json(200, result)
        else:
            self.send_json(503, {"error": "cluster manager not enabled"})

    def handle_peers(self) -> None:
        """Handle ``GET /peers``."""
        if self.server.cluster_manager:
            peers = self.server.cluster_manager.get_peers()
            self.send_json(200, {"peers": peers})
        else:
            self.send_json(503, {"error": "cluster manager not enabled"})


class HTTPServer:
    """Production HTTP server wrapping a MembraneNode.

    Args:
        node: MembraneNode to serve.
        host: Bind address.
        port: Listen port.
        compute_backend: Optional compute backend for prefill.
        transfer_service: Optional transfer service for sync.
        cluster_manager: Optional cluster manager for peer
            management.
    """

    def __init__(
        self,
        node: MembraneNode,
        host: str = "0.0.0.0",
        port: int = 8080,
        compute_backend: ComputeBackend | None = None,
        transfer_service: TransferService | None = None,
        cluster_manager: Any | None = None,
    ) -> None:
        """Initialize the HTTP server wrapper.

        Args:
            node: MembraneNode to serve.
            host: Bind address.
            port: Listen port.
            compute_backend: Optional compute backend.
            transfer_service: Optional transfer service. A
                default :class:`TransferService` is used when
                ``None``.
            cluster_manager: Optional cluster manager.
        """
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service or TransferService()
        self.cluster_manager = cluster_manager
        self._server: MembraneStdlibHTTPServer | None = None

    def start(self) -> None:
        """Start the HTTP server (blocking).

        Calls :meth:`http.server.HTTPServer.serve_forever` on
        the underlying server. The caller is expected to invoke
        :meth:`stop` from another thread to terminate the
        serve loop.
        """
        self._server = MembraneStdlibHTTPServer(
            (self.host, self.port),
            MembraneHTTPHandler,
            node=self.node,
            compute_backend=self.compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=self.cluster_manager,
        )
        logger.info("HTTP server listening on http://%s:%s", self.host, self.port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server.

        No-op when the server was never started.
        """
        if self._server:
            self._server.shutdown()
            logger.info("HTTP server stopped")

    def run_in_thread(self) -> None:
        """Start the server in a background daemon thread.

        Useful when the caller's main thread should keep doing
        other work (e.g., running the CLI or the TUI dashboard).
        """
        import threading

        t = threading.Thread(target=self.start, daemon=True)
        t.start()
