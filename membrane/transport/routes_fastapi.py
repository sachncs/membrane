"""FastAPI route handlers.

Module-level functions that wire Membrane's business logic to FastAPI
endpoints. The corresponding stdlib HTTP version lives in
:mod:`membrane.transport.routes` and shares the same URL contract
and semantics; this module is the async-friendly binding.

Pydantic models at the top of the module describe the request bodies.
Each handler is registered via :func:`register_routes`, which is the
single place to add or remove endpoints.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from membrane.compute.cpu import CPU
from membrane.serialization import from_dict as deserialize_fragment
from membrane.serialization import to_dict as serialize_fragment

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic request models
# ------------------------------------------------------------------


import json


class FragmentPayload(BaseModel):
    """Wire format for a serialized Fragment.

    Mirrors the canonical schema in :mod:`membrane.serialization`:
    layer/token bounds are flattened (``layer_start``, ``layer_end``,
    ``token_start``, ``token_end``) and the embedding is a JSON-encoded
    string for compactness over the wire.
    """

    schema_version: int = 1
    content_hash: str
    embedding: str  # JSON-encoded list[float]
    model_id: str
    layer_start: int
    layer_end: int
    token_start: int
    token_end: int
    size: int
    ttl: float
    reuse_score: float
    version_id: int

    def to_wire_dict(self) -> dict[str, Any]:
        """Transform into the canonical wire dict for ``from_dict``."""
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "embedding": self.embedding,  # already JSON-encoded
            "model_id": self.model_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "size": self.size,
            "ttl": self.ttl,
            "reuse_score": self.reuse_score,
            "version_id": self.version_id,
        }


class StoreRequest(BaseModel):
    """``POST /store`` body."""

    fragment: FragmentPayload
    is_primary: bool = False


class ReplicateRequest(BaseModel):
    """``POST /replicate`` body."""

    fragment: FragmentPayload


class PrefillRequest(BaseModel):
    """``POST /prefill`` body."""

    prompt_tokens: list[int]
    model_id: str = "default"


class SyncRequest(BaseModel):
    """``POST /sync`` body."""

    source_url: str


class JoinRequest(BaseModel):
    """``POST /join`` body."""

    node_id: str
    host: str
    port: int


class LeaveRequest(BaseModel):
    """``POST /leave`` body."""

    node_id: str


class GossipRequest(BaseModel):
    """``POST /gossip`` body."""

    peers: list[dict[str, Any]] = []
    fragment_locations: dict[str, list[str]] = {}
    inventory_digest: dict[str, int] = {}


# ------------------------------------------------------------------
# Endpoint handlers
# ------------------------------------------------------------------


def heartbeat(app: FastAPI) -> dict[str, Any]:
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


def livez(_app: FastAPI) -> dict[str, str]:
    """``GET /livez`` — liveness probe."""
    return {"status": "alive"}


def readyz(app: FastAPI):
    """``GET /readyz`` — readiness probe (deep)."""
    if not app.state.node:
        return JSONResponse({"status": "no node"}, status_code=503)
    stats = app.state.node.get_stats()
    if stats.memory_used_bytes >= stats.memory_limit_bytes:
        return JSONResponse(
            {"status": "memory saturated", "memory_used_bytes": stats.memory_used_bytes},
            status_code=503,
        )
    return {"status": "ready"}


def metrics(app: FastAPI):
    """``GET /metrics`` — Prometheus text exposition."""
    if app.state.metrics_registry is not None:
        return PlainTextResponse(
            app.state.metrics_registry.render(),
            media_type="text/plain; version=0.0.4",
        )
    # Legacy JSON fallback when no registry was supplied.
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


def metrics_json(app: FastAPI) -> dict[str, Any]:
    """``GET /metrics.json`` — legacy JSON snapshot for the TUI."""
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


def retrieve(app: FastAPI, content_hash: str) -> dict[str, Any]:
    """``GET /retrieve?content_hash=...``."""
    if not app.state.node:
        return {"found": False, "fragment": None}
    frag = app.state.node.retrieve(content_hash)
    if frag:
        return {"found": True, "fragment": serialize_fragment(frag)}
    return {"found": False, "fragment": None}


def inventory(app: FastAPI) -> dict[str, Any]:
    """``GET /inventory`` — node's inventory digest."""
    if not app.state.node:
        return {"node_id": "", "digest": {}}
    digest = {h: frag.version_id for h, frag in app.state.node.fragments.items()}
    return {"node_id": app.state.node.node_id, "digest": digest}


def peers(app: FastAPI) -> dict[str, Any]:
    """``GET /peers`` — cluster membership view."""
    if app.state.cluster_manager:
        return {"peers": app.state.cluster_manager.get_peers()}
    return {"error": "cluster manager not enabled"}


def store(app: FastAPI, req: StoreRequest) -> dict[str, Any]:
    """``POST /store``."""
    try:
        frag = deserialize_fragment(req.fragment.to_wire_dict())
        ok = app.state.node.store(frag, is_primary=req.is_primary) if app.state.node else False
        return {"success": ok, "content_hash": frag.content_hash}
    except Exception:
        logger.exception("store failed")
        return {"error": "internal"}


def replicate(app: FastAPI, req: ReplicateRequest) -> dict[str, Any]:
    """``POST /replicate`` — store a fragment as a non-primary replica."""
    try:
        frag = deserialize_fragment(req.fragment.to_wire_dict())
        ok = app.state.node.store(frag, is_primary=False) if app.state.node else False
        return {"success": ok, "content_hash": frag.content_hash}
    except Exception:
        logger.exception("replicate failed")
        return {"error": "internal"}


def sync(app: FastAPI, req: SyncRequest) -> dict[str, Any]:
    """``POST /sync`` — pull missing fragments from a source URL."""
    source_url = req.source_url
    if not source_url:
        return {"error": "missing source_url"}
    try:
        with urlopen(Request(f"{source_url}/inventory"), timeout=5) as resp:  # noqa: S310
            remote_data = json.loads(resp.read().decode())
        remote_digest = remote_data.get("digest", {})
        local_digest = app.state.transfer_service.inventory_digest(app.state.node)
        missing = app.state.transfer_service.compare_inventories(local_digest, remote_digest)
        transferred: list[str] = []
        for h in missing:
            with urlopen(Request(f"{source_url}/retrieve?content_hash={h}"), timeout=5) as resp:  # noqa: S310
                remote_frag_data = json.loads(resp.read().decode())
            if remote_frag_data.get("found"):
                frag = deserialize_fragment(remote_frag_data["fragment"])
                if app.state.node.store(frag, is_primary=False):
                    transferred.append(h)
        return {"success": True, "transferred": transferred}
    except Exception:
        logger.exception("sync failed")
        return {"error": "internal"}


def prefill(app: FastAPI, req: PrefillRequest) -> dict[str, Any]:
    """``POST /prefill`` — run prefill and store fragments as primary."""
    backend = app.state.compute_backend or CPU()
    try:
        fragments = backend.prefill(req.prompt_tokens, req.model_id)
        for frag in fragments:
            if app.state.node:
                app.state.node.store(frag, is_primary=True)
        return {
            "success": True,
            "fragments": [serialize_fragment(f) for f in fragments],
        }
    except Exception:
        logger.exception("prefill failed")
        return {"error": "internal"}


def join(app: FastAPI, req: JoinRequest) -> dict[str, Any]:
    """``POST /join``."""
    if not req.node_id or not req.host or not req.port:
        return {"error": "missing node_id, host, or port"}
    if app.state.cluster_manager:
        return app.state.cluster_manager.on_peer_join(req.node_id, req.host, req.port)
    return {"error": "cluster manager not enabled"}


def leave(app: FastAPI, req: LeaveRequest) -> dict[str, Any]:
    """``POST /leave``."""
    if not req.node_id:
        return {"error": "missing node_id"}
    if app.state.cluster_manager:
        app.state.cluster_manager.on_peer_leave(req.node_id)
        return {"success": True}
    return {"error": "cluster manager not enabled"}


def gossip(app: FastAPI, req: GossipRequest) -> dict[str, Any]:
    """``POST /gossip``."""
    if app.state.cluster_manager:
        return app.state.cluster_manager.on_gossip(req.model_dump())
    return {"error": "cluster manager not enabled"}


# ------------------------------------------------------------------
# Route registration
# ------------------------------------------------------------------


def register_routes(app: FastAPI) -> None:
    """Register every Membrane HTTP route on ``app``.

    This is the single place to add or remove endpoints. Each route
    is paired with its handler above; the registration glue is here
    so that a single change to a URL or method is local.

    Args:
        app: The FastAPI app to configure.
    """
    # GET endpoints
    app.add_api_route("/heartbeat", lambda: heartbeat(app), methods=["GET"])
    app.add_api_route("/livez", lambda: livez(app), methods=["GET"])
    app.add_api_route("/readyz", lambda: readyz(app), methods=["GET"])
    app.add_api_route("/metrics", lambda: metrics(app), methods=["GET"])
    app.add_api_route("/metrics.json", lambda: metrics_json(app), methods=["GET"])

    def retrieve_handler(content_hash: str):
        return retrieve(app, content_hash)

    app.add_api_route("/retrieve", retrieve_handler, methods=["GET"])
    app.add_api_route("/inventory", lambda: inventory(app), methods=["GET"])
    app.add_api_route("/peers", lambda: peers(app), methods=["GET"])

    # POST endpoints. Use module-level factories (closures) that capture
    # ``app`` so the route functions take only the request body, which
    # is the only FastAPI parameter it needs.
    def store_handler(req: StoreRequest):
        return store(app, req)

    def replicate_handler(req: ReplicateRequest):
        return replicate(app, req)

    def sync_handler(req: SyncRequest):
        return sync(app, req)

    def prefill_handler(req: PrefillRequest):
        return prefill(app, req)

    def join_handler(req: JoinRequest):
        return join(app, req)

    def leave_handler(req: LeaveRequest):
        return leave(app, req)

    def gossip_handler(req: GossipRequest):
        return gossip(app, req)

    app.add_api_route("/store", store_handler, methods=["POST"], response_model=None)
    app.add_api_route("/replicate", replicate_handler, methods=["POST"], response_model=None)
    app.add_api_route("/sync", sync_handler, methods=["POST"], response_model=None)
    app.add_api_route("/prefill", prefill_handler, methods=["POST"], response_model=None)
    app.add_api_route("/join", join_handler, methods=["POST"], response_model=None)
    app.add_api_route("/leave", leave_handler, methods=["POST"], response_model=None)
    app.add_api_route("/gossip", gossip_handler, methods=["POST"], response_model=None)


__all__ = ["register_routes"]
