"""FastAPIServer: production HTTP/REST server using FastAPI + uvicorn.

Replaces the stdlib :mod:`http.server` for high-RPS async
serving. All endpoints defined by
:class:`~membrane.transport.http_server.HTTPServer` are preserved
with identical request/response shapes.

Pydantic models at the top of the module describe the request
bodies; the application factory :func:`create_app` registers the
endpoints and attaches the supplied :class:`MembraneNode`,
:class:`ComputeBackend`, :class:`TransferService`, and optional
cluster manager to ``app.state`` for the handlers to consume.

Security:
    * The server is unauthenticated. Place it behind an
      authenticating reverse proxy in production.
"""

import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from membrane.compute.backend import ComputeBackend
from membrane.compute.cpu_backend import CPUBackend
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature
from membrane.transfer_service import TransferService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class FragmentPayload(BaseModel):
    """Pydantic representation of a :class:`Fragment` for HTTP transport.

    Attributes:
        content_hash: Content hash identifying the fragment.
        embedding: Dense embedding (converted to a tuple on
            deserialization).
        model_id: Model identifier.
        layer_range: ``[start, end]`` layer bounds.
        token_span: ``[start, end]`` token position bounds.
        size: Payload size in bytes.
        ttl: Time-to-live in seconds.
        reuse_score: Reuse likelihood in ``[0, 1]``.
        version_id: Monotonic version counter.
    """

    content_hash: str
    embedding: list[float]
    model_id: str
    layer_range: list[int]
    token_span: list[int]
    size: int
    ttl: float
    reuse_score: float
    version_id: int


class StoreRequest(BaseModel):
    """Request body for ``POST /store``.

    Attributes:
        fragment: Fragment to store.
        is_primary: Whether the local node should claim primary
            ownership.
    """

    fragment: FragmentPayload
    is_primary: bool = False


class ReplicateRequest(BaseModel):
    """Request body for ``POST /replicate``.

    Attributes:
        fragment: Fragment to replicate.
    """

    fragment: FragmentPayload


class PrefillRequest(BaseModel):
    """Request body for ``POST /prefill``.

    Attributes:
        prompt_tokens: Input token IDs.
        model_id: Model identifier. Defaults to ``"default"``.
    """

    prompt_tokens: list[int]
    model_id: str = "default"


class SyncRequest(BaseModel):
    """Request body for ``POST /sync``.

    Attributes:
        source_url: Base URL of the remote Membrane node to
            pull missing fragments from.
    """

    source_url: str


class JoinRequest(BaseModel):
    """Request body for ``POST /join``.

    Attributes:
        node_id: Joining node's identifier.
        host: Joining node's host.
        port: Joining node's port.
    """

    node_id: str
    host: str
    port: int


class LeaveRequest(BaseModel):
    """Request body for ``POST /leave``.

    Attributes:
        node_id: Leaving node's identifier.
    """

    node_id: str


class GossipRequest(BaseModel):
    """Request body for ``POST /gossip``.

    Attributes:
        node_id: Sender's identifier.
        timestamp: Sender's wall-clock time.
        peers: List of known peers.
        fragment_locations: Sampled fragment-location map.
        inventory_digest: ``content_hash -> version_id`` for
            fragments the sender holds locally.
    """

    node_id: str
    timestamp: float
    peers: list[dict[str, Any]] = []
    fragment_locations: dict[str, list[str]] = {}
    inventory_digest: dict[str, int] = {}


# ------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------

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


def deserialize_fragment(data: FragmentPayload) -> Fragment:
    """Reconstruct a fragment from a Pydantic payload.

    Args:
        data: ``FragmentPayload`` from the request body.

    Returns:
        Fragment: Reconstructed fragment instance.
    """
    return Fragment(
        content_hash=data.content_hash,
        embedding=tuple(data.embedding),
        structural_signature=StructuralSignature(
            model_id=data.model_id,
            layer_range=(data.layer_range[0], data.layer_range[1]),
            token_span=(data.token_span[0], data.token_span[1]),
        ),
        size=data.size,
        ttl=data.ttl,
        reuse_score=data.reuse_score,
        version_id=data.version_id,
    )


# ------------------------------------------------------------------
# FastAPI application factory
# ------------------------------------------------------------------

def create_app(
    node: MembraneNode,
    compute_backend: ComputeBackend | None,
    transfer_service: TransferService,
    cluster_manager: Any | None,
) -> FastAPI:
    """Build a configured FastAPI application for a Membrane node.

    Args:
        node: Local :class:`MembraneNode`.
        compute_backend: Optional :class:`ComputeBackend`.
        transfer_service: :class:`TransferService`.
        cluster_manager: Optional cluster manager.

    Returns:
        FastAPI: Configured application ready to be served by
        uvicorn.
    """
    app = FastAPI(title="Membrane", version="0.1.0")
    app.state.node = node
    app.state.compute_backend = compute_backend
    app.state.transfer_service = transfer_service
    app.state.cluster_manager = cluster_manager

    # ------------------------------------------------------------------
    # GET endpoints
    # ------------------------------------------------------------------

    @app.get("/heartbeat")
    def heartbeat() -> dict[str, Any]:
        """``GET /heartbeat`` — node health and load snapshot."""
        if not app.state.node:
            return {"error": "no node"}
        stats = app.state.node.get_stats()
        return {
            "node_id": app.state.node.node_id,
            "load": app.state.node.heartbeat(),
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "healthy": True,
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        """``GET /metrics`` — extended metrics payload."""
        if not app.state.node:
            return {"error": "no node"}
        stats = app.state.node.get_stats()
        return {
            "node_id": app.state.node.node_id,
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "load": app.state.node.heartbeat(),
        }

    @app.get("/retrieve")
    def retrieve(content_hash: str) -> dict[str, Any]:
        """``GET /retrieve?content_hash=...``."""
        if not app.state.node:
            return {"found": False, "fragment": None}
        frag = app.state.node.retrieve(content_hash)
        if frag:
            return {"found": True, "fragment": serialize_fragment(frag)}
        return {"found": False, "fragment": None}

    @app.get("/inventory")
    def inventory() -> dict[str, Any]:
        """``GET /inventory`` — return the node's inventory digest."""
        if not app.state.node:
            return {"node_id": "", "digest": {}}
        digest = {h: frag.version_id for h, frag in app.state.node.fragments.items()}
        return {"node_id": app.state.node.node_id, "digest": digest}

    @app.get("/peers")
    def peers() -> dict[str, Any]:
        """``GET /peers`` — return the cluster membership view."""
        if app.state.cluster_manager:
            return {"peers": app.state.cluster_manager.get_peers()}
        return {"error": "cluster manager not enabled"}

    # ------------------------------------------------------------------
    # POST endpoints
    # ------------------------------------------------------------------

    @app.post("/store")
    def store(req: StoreRequest) -> dict[str, Any]:
        """``POST /store`` — store a fragment on the local node."""
        try:
            frag = deserialize_fragment(req.fragment)
            ok = (
                app.state.node.store(frag, is_primary=req.is_primary)
                if app.state.node
                else False
            )
            return {"success": ok, "content_hash": frag.content_hash}
        except Exception as exc:
            logger.exception("store failed")
            return {"error": str(exc)}

    @app.post("/replicate")
    def replicate(req: ReplicateRequest) -> dict[str, Any]:
        """``POST /replicate`` — store a fragment as a non-primary replica."""
        try:
            frag = deserialize_fragment(req.fragment)
            ok = app.state.node.store(frag, is_primary=False) if app.state.node else False
            return {"success": ok, "content_hash": frag.content_hash}
        except Exception as exc:
            logger.exception("replicate failed")
            return {"error": str(exc)}

    @app.post("/sync")
    def sync(req: SyncRequest) -> dict[str, Any]:
        """``POST /sync`` — pull missing fragments from a source URL."""
        source_url = req.source_url
        if not source_url:
            return {"error": "missing source_url"}
        try:
            import json
            import urllib.request

            # Pull remote inventory.
            inv_req = urllib.request.Request(f"{source_url}/inventory")
            with urllib.request.urlopen(inv_req, timeout=5) as resp:
                remote_data = json.loads(resp.read().decode())
            remote_digest = remote_data.get("digest", {})
            local_digest = app.state.transfer_service.inventory_digest(app.state.node)
            missing = app.state.transfer_service.compare_inventories(local_digest, remote_digest)
            transferred: list[str] = []
            for h in missing:
                ret_req = urllib.request.Request(f"{source_url}/retrieve?content_hash={h}")
                with urllib.request.urlopen(ret_req, timeout=5) as resp:
                    remote_frag_data = json.loads(resp.read().decode())
                if remote_frag_data.get("found"):
                    frag = deserialize_fragment(FragmentPayload(**remote_frag_data["fragment"]))
                    if app.state.node.store(frag, is_primary=False):
                        transferred.append(h)
            return {"success": True, "transferred": transferred}
        except Exception as exc:
            logger.exception("sync failed")
            return {"error": str(exc)}

    @app.post("/prefill")
    def prefill(req: PrefillRequest) -> dict[str, Any]:
        """``POST /prefill`` — run prefill and store fragments as primary."""
        backend = app.state.compute_backend or CPUBackend()
        try:
            fragments = backend.prefill(req.prompt_tokens, req.model_id)
            for frag in fragments:
                if app.state.node:
                    app.state.node.store(frag, is_primary=True)
            return {
                "success": True,
                "fragments": [serialize_fragment(f) for f in fragments],
            }
        except Exception as exc:
            logger.exception("prefill failed")
            return {"error": str(exc)}

    @app.post("/join")
    def join(req: JoinRequest) -> dict[str, Any]:
        """``POST /join`` — bootstrap a new peer into the cluster."""
        if not req.node_id or not req.host or not req.port:
            return {"error": "missing node_id, host, or port"}
        if app.state.cluster_manager:
            return app.state.cluster_manager.on_peer_join(req.node_id, req.host, req.port)
        return {"error": "cluster manager not enabled"}

    @app.post("/leave")
    def leave(req: LeaveRequest) -> dict[str, Any]:
        """``POST /leave`` — remove a peer from the cluster."""
        if not req.node_id:
            return {"error": "missing node_id"}
        if app.state.cluster_manager:
            app.state.cluster_manager.on_peer_leave(req.node_id)
            return {"success": True}
        return {"error": "cluster manager not enabled"}

    @app.post("/gossip")
    def gossip(req: GossipRequest) -> dict[str, Any]:
        """``POST /gossip`` — exchange gossip state with the cluster."""
        if app.state.cluster_manager:
            return app.state.cluster_manager.on_gossip(req.model_dump())
        return {"error": "cluster manager not enabled"}

    return app


# ------------------------------------------------------------------
# Server wrapper
# ------------------------------------------------------------------

class FastAPIServer:
    """Production HTTP server using FastAPI + uvicorn.

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
        """Initialize the FastAPI server wrapper."""
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.transfer_service = transfer_service or TransferService()
        self.cluster_manager = cluster_manager
        self.app = create_app(
            node=node,
            compute_backend=compute_backend,
            transfer_service=self.transfer_service,
            cluster_manager=cluster_manager,
        )
        self._server: Any | None = None

    def start(self) -> None:
        """Start the uvicorn server (blocking).

        Calls :meth:`uvicorn.Server.run` which blocks until
        :attr:`should_exit` is set (typically by :meth:`stop`
        from another thread).
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
