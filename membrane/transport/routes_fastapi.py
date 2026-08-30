"""FastAPI route bindings.

This module binds the operations in :mod:`membrane.transport.ops`
to FastAPI endpoints. The Pydantic models at the top describe the
request bodies; the handlers are thin shims that pass the validated
request into the corresponding operation.

The stdlib HTTP version lives in :mod:`membrane.transport.routes`
and shares the same URL contract. Both transports delegate to the
shared operations so the actual store / retrieve / sync logic
lives in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from membrane.compute.cpu import CPU
from membrane.serialization import JsonDict
from membrane.transport.ops import (
    MAX_BODY_BYTES,
    op_gossip,
    op_heartbeat,
    op_inventory,
    op_join,
    op_leave,
    op_metrics,
    op_peers,
    op_prefill,
    op_replicate,
    op_retrieve,
    op_store,
    op_sync,
)

logger = logging.getLogger(__name__)


def _authenticator_for(app: FastAPI) -> object | None:
    """Return the cluster's authenticator from app.state, if any.

    When ``app.state.authenticator`` was populated by the Server at
    startup with an
    :class:`~membrane.auth.mtls.MTLSAuthenticator`, ``op_join``
    and ``op_heartbeat`` consume it. Absent the cluster is treated
    as non-mTLS (single-node deployments).
    """
    return getattr(app.state, "authenticator", None)


def _peer_headers(request: Request) -> dict[str, str]:
    """Capture inbound headers as a plain dict for op_join / op_heartbeat.

    FastAPI's :class:`Request.headers` is case-insensitive; we
    downcase keys here so
    :meth:`membrane.transport.tls.parse_peer_cn_header` reads them
    with the ``x-ssl-client-cn`` spelling it expects.
    """
    return {k.lower(): v for k, v in request.headers.items()}


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class FragmentPayload(BaseModel):
    """Wire format for a serialized Fragment.

    Mirrors the 2.0 canonical schema in
    :mod:`membrane.serialization`. The :class:`~membrane.identity.PayloadIdentity`
    is carried as a nested ``identity`` object; ranges are JSON
    arrays of two ints; ``shape`` is a list of ints; ``consistency``
    is one of strong / quorum / eventual; ``hlc`` is the wire
    integer from :class:`~membrane.hlc.HLC`.
    """

    schema_version: int = 3
    identity: dict[str, Any]
    payload_ref: str | None = None
    payload_size: int
    ttl: float
    reuse_score: float
    version_id: int
    consistency: str = "strong"
    hlc: int = 0

    def to_wire_dict(self) -> JsonDict:
        """Transform into the canonical wire dict accepted by
        :func:`membrane.serialization.from_dict`."""
        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "payload_ref": self.payload_ref,
            "payload_size": self.payload_size,
            "ttl": self.ttl,
            "reuse_score": self.reuse_score,
            "version_id": self.version_id,
            "consistency": self.consistency,
            "hlc": self.hlc,
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


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


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
            {
                "status": "memory saturated",
                "memory_used_bytes": stats.memory_used_bytes,
            },
            status_code=503,
        )
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_routes(app: FastAPI) -> None:
    """Register every Membrane HTTP route on ``app``.

    Each route delegates to the corresponding operation in
    :mod:`membrane.transport.ops`. The Pydantic models at the top
    of this module validate request bodies; the inline lambdas
    capture ``app`` so the registered functions take only the
    request body.

    Args:
        app: The FastAPI app to configure.
    """
    # GET endpoints that have no body.
    def heartbeat_handler(request: Request):
        return _heartbeat(app, request)

    app.add_api_route("/heartbeat", heartbeat_handler, methods=["GET"])
    app.add_api_route("/livez", lambda: livez(app), methods=["GET"])
    app.add_api_route("/readyz", lambda: readyz(app), methods=["GET"])
    app.add_api_route("/metrics", lambda: _metrics(app), methods=["GET"])
    app.add_api_route("/metrics.json", lambda: _metrics_json(app), methods=["GET"])
    app.add_api_route(
        "/retrieve",
        lambda content_hash: _retrieve(app, content_hash),
        methods=["GET"],
    )
    app.add_api_route("/inventory", lambda: _inventory(app), methods=["GET"])
    app.add_api_route("/peers", lambda: _peers(app), methods=["GET"])

    # POST endpoints.
    def store_handler(req: StoreRequest):
        return _store(app, req)

    def replicate_handler(req: ReplicateRequest):
        return _replicate(app, req)

    def sync_handler(req: SyncRequest):
        return _sync(app, req)

    def prefill_handler(req: PrefillRequest):
        return _prefill(app, req)

    def join_handler(req: JoinRequest, request: Request):
        return _join(app, req, request)

    def leave_handler(req: LeaveRequest):
        return _leave(app, req)

    def gossip_handler(req: GossipRequest):
        return _gossip(app, req)

    app.add_api_route("/store", store_handler, methods=["POST"], response_model=None)
    app.add_api_route("/replicate", replicate_handler, methods=["POST"], response_model=None)
    app.add_api_route("/sync", sync_handler, methods=["POST"], response_model=None)
    app.add_api_route("/prefill", prefill_handler, methods=["POST"], response_model=None)
    app.add_api_route("/join", join_handler, methods=["POST"], response_model=None)
    app.add_api_route("/leave", leave_handler, methods=["POST"], response_model=None)
    app.add_api_route("/gossip", gossip_handler, methods=["POST"], response_model=None)


def _heartbeat(app: FastAPI, request: Request):
    status, body = op_heartbeat(
        app.state.node,
        cluster=getattr(app.state, "cluster_manager", None),
        headers=_peer_headers(request),
    )
    return _respond(status, body)


def _metrics(app: FastAPI):
    """``GET /metrics`` — Prometheus text or legacy JSON."""
    status, payload = op_metrics(app.state.node, app.state.metrics_registry)
    if status == 200 and isinstance(payload, tuple):
        text, headers = payload
        return PlainTextResponse(text, media_type=headers["media_type"])
    return _respond(status, payload)


def _metrics_json(app: FastAPI):
    """``GET /metrics.json`` — legacy JSON snapshot for the TUI."""
    status, body = op_metrics(app.state.node)
    return _respond(status, body)


def _inventory(app: FastAPI):
    status, body = op_inventory(app.state.node)
    return _respond(status, body)


def _peers(app: FastAPI):
    status, body = op_peers(app.state.cluster_manager)
    return _respond(status, body)


def _retrieve(app: FastAPI, content_hash: str):
    status, body = op_retrieve(app.state.node, content_hash)
    return _respond(status, body)


def _store(app: FastAPI, req: StoreRequest):
    status, body = op_store(
        app.state.node,
        req.fragment.to_wire_dict(),
        req.is_primary,
        cluster=getattr(app.state, "cluster_manager", None),
        quorum_attempt=getattr(app.state, "quorum_attempt", None),
        draining=bool(getattr(app.state.server, "is_draining", False))
        if hasattr(app.state, "server")
        else False,
    )
    return _respond(status, body)


def _respond_with_retry_after(status: int, body: JsonDict, retry_after: int) -> Response:
    return JSONResponse(
        body,
        status_code=status,
        headers={"Retry-After": str(retry_after)},
    )


def _replicate(app: FastAPI, req: ReplicateRequest):
    status, body = op_replicate(
        app.state.node,
        req.fragment.to_wire_dict(),
    )
    return _respond(status, body)


def _sync(app: FastAPI, req: SyncRequest):
    status, body = op_sync(app.state.node, app.state.transfer_service, req.source_url)
    return _respond(status, body)


def _prefill(app: FastAPI, req: PrefillRequest):
    status, body = op_prefill(
        app.state.node,
        app.state.compute_backend or CPU(),
        req.prompt_tokens,
        req.model_id,
    )
    return _respond(status, body)


def _join(app: FastAPI, req: JoinRequest, request: Request):
    status, body = op_join(
        app.state.cluster_manager,
        req.node_id,
        req.host,
        req.port,
        headers=_peer_headers(request),
        authenticator=_authenticator_for(app),
    )
    return _respond(status, body)


def _leave(app: FastAPI, req: LeaveRequest):
    status, body = op_leave(app.state.cluster_manager, req.node_id)
    return _respond(status, body)


def _gossip(app: FastAPI, req: GossipRequest):
    status, body = op_gossip(app.state.cluster_manager, req.model_dump())
    return _respond(status, body)


def _respond(status: int, body: JsonDict | tuple[str, dict[str, str]] | object) -> Response:
    """Translate an operation's ``(status, body)`` to a FastAPI response."""
    if status == 200:
        return JSONResponse(body)
    return JSONResponse(body, status_code=status)


__all__ = ["MAX_BODY_BYTES", "register_routes"]
