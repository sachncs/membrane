"""HTTPServer: production HTTP/REST server for Membrane nodes.

Uses Python stdlib ``http.server`` so there are zero external dependencies
for the HTTP transport.  Supports JSON request/response.

Endpoints:
  POST /store       -> Store a fragment
  GET  /retrieve    -> Retrieve a fragment by content_hash
  GET  /inventory   -> Get node inventory digest
  POST /sync        -> Sync fragments from source node
  GET  /heartbeat   -> Node health and load
  POST /prefill     -> Run prefill and return fragments
  POST /join        -> Join the cluster
  POST /leave       -> Leave the cluster
  POST /gossip      -> Exchange gossip state
  GET  /peers       -> List known peers
  POST /replicate   -> Store a fragment as replica
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


def _serialize_fragment(frag: Fragment) -> dict[str, Any]:
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


def _deserialize_fragment(data: dict[str, Any]) -> Fragment:
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


class _MembraneServer(StdlibHTTPServer):
    """Custom HTTPServer that holds references to node, compute, and cluster."""

    def __init__(
        self,
        server_address,
        handler_class,
        node: MembraneNode,
        compute_backend: ComputeBackend | None,
        transfer_service: TransferService,
        cluster_manager: Any | None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.node = node
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service
        self.cluster_manager = cluster_manager


class _MembraneHTTPHandler(BaseHTTPRequestHandler):
    """Request handler for Membrane HTTP transport."""

    server: _MembraneServer  # type: ignore[misc]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body.decode()) if body else {}

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/retrieve":
            self._handle_retrieve()
        elif path == "/inventory":
            self._handle_inventory()
        elif path == "/heartbeat":
            self._handle_heartbeat()
        elif path == "/metrics":
            self._handle_metrics()
        elif path == "/peers":
            self._handle_peers()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/store":
            self._handle_store()
        elif path == "/sync":
            self._handle_sync()
        elif path == "/prefill":
            self._handle_prefill()
        elif path == "/join":
            self._handle_join()
        elif path == "/leave":
            self._handle_leave()
        elif path == "/gossip":
            self._handle_gossip()
        elif path == "/replicate":
            self._handle_replicate()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_store(self) -> None:
        data = self._read_json()
        try:
            frag = _deserialize_fragment(data["fragment"])
            is_primary = data.get("is_primary", False)
            ok = self.server.node.store(frag, is_primary=is_primary) if self.server.node else False
            self._send_json(200 if ok else 500, {"success": ok, "content_hash": frag.content_hash})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_replicate(self) -> None:
        data = self._read_json()
        try:
            frag = _deserialize_fragment(data["fragment"])
            ok = self.server.node.store(frag, is_primary=False) if self.server.node else False
            self._send_json(200 if ok else 500, {"success": ok, "content_hash": frag.content_hash})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_retrieve(self) -> None:
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(self.path).query)
        h = qs.get("content_hash", [None])[0]
        if not h:
            self._send_json(400, {"error": "missing content_hash"})
            return
        frag = self.server.node.retrieve(h) if self.server.node else None
        if frag:
            self._send_json(200, {"found": True, "fragment": _serialize_fragment(frag)})
        else:
            self._send_json(404, {"found": False, "fragment": None})

    def _handle_inventory(self) -> None:
        digest = {h: frag.version_id for h, frag in (self.server.node.fragments.items() if self.server.node else {})}
        self._send_json(200, {"node_id": self.server.node.node_id if self.server.node else "", "digest": digest})

    def _handle_heartbeat(self) -> None:
        if not self.server.node:
            self._send_json(500, {"error": "no node"})
            return
        stats = self.server.node.get_stats()
        self._send_json(200, {
            "node_id": self.server.node.node_id,
            "load": self.server.node.heartbeat(),
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "healthy": True,
        })

    def _handle_metrics(self) -> None:
        if not self.server.node:
            self._send_json(500, {"error": "no node"})
            return
        stats = self.server.node.get_stats()
        self._send_json(200, {
            "node_id": self.server.node.node_id,
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "load": self.server.node.heartbeat(),
        })

    def _handle_sync(self) -> None:
        data = self._read_json()
        source_url = data.get("source_url", "")
        if not source_url:
            self._send_json(400, {"error": "missing source_url"})
            return
        try:
            # Pull inventory from remote and transfer missing fragments
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
                    frag = _deserialize_fragment(remote_frag_data["fragment"])
                    if self.server.node.store(frag, is_primary=False):
                        transferred.append(h)
            self._send_json(200, {"success": True, "transferred": transferred})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_prefill(self) -> None:
        data = self._read_json()
        tokens = data.get("prompt_tokens", [])
        model_id = data.get("model_id", "default")
        backend = self.server.compute_backend or CPUBackend()
        try:
            fragments = backend.prefill(tokens, model_id)
            for frag in fragments:
                if self.server.node:
                    self.server.node.store(frag, is_primary=True)
            self._send_json(200, {
                "success": True,
                "fragments": [_serialize_fragment(f) for f in fragments],
            })
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_join(self) -> None:
        data = self._read_json()
        node_id = data.get("node_id", "")
        host = data.get("host", "")
        port = data.get("port", 0)
        if not node_id or not host or not port:
            self._send_json(400, {"error": "missing node_id, host, or port"})
            return
        if self.server.cluster_manager:
            result = self.server.cluster_manager.on_peer_join(node_id, host, port)
            self._send_json(200, result)
        else:
            self._send_json(503, {"error": "cluster manager not enabled"})

    def _handle_leave(self) -> None:
        data = self._read_json()
        node_id = data.get("node_id", "")
        if not node_id:
            self._send_json(400, {"error": "missing node_id"})
            return
        if self.server.cluster_manager:
            self.server.cluster_manager.on_peer_leave(node_id)
            self._send_json(200, {"success": True})
        else:
            self._send_json(503, {"error": "cluster manager not enabled"})

    def _handle_gossip(self) -> None:
        data = self._read_json()
        if self.server.cluster_manager:
            result = self.server.cluster_manager.on_gossip(data)
            self._send_json(200, result)
        else:
            self._send_json(503, {"error": "cluster manager not enabled"})

    def _handle_peers(self) -> None:
        if self.server.cluster_manager:
            peers = self.server.cluster_manager.get_peers()
            self._send_json(200, {"peers": peers})
        else:
            self._send_json(503, {"error": "cluster manager not enabled"})


class HTTPServer:
    """Production HTTP server wrapping a MembraneNode.

    Args:
        node: MembraneNode to serve.
        host: Bind address.
        port: Listen port.
        compute_backend: Optional compute backend for prefill.
        transfer_service: Optional transfer service for sync.
        cluster_manager: Optional cluster manager for peer management.
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
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service or TransferService()
        self.cluster_manager = cluster_manager
        self._server: _MembraneServer | None = None

    def start(self) -> None:
        """Start the HTTP server (blocking)."""
        self._server = _MembraneServer(
            (self.host, self.port),
            _MembraneHTTPHandler,
            node=self.node,
            compute_backend=self.compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=self.cluster_manager,
        )
        logger.info("HTTP server listening on http://%s:%s", self.host, self.port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            logger.info("HTTP server stopped")

    def run_in_thread(self) -> None:
        """Start the server in a background daemon thread."""
        import threading

        t = threading.Thread(target=self.start, daemon=True)
        t.start()
