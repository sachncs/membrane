"""FastAPI route bindings.

This module binds the operations in :mod:`membrane.transport.ops`
to FastAPI endpoints. The Pydantic models at the top describe the
request bodies; the handlers delegate to the corresponding
operation after running the per-route scope check from
:mod:`membrane.transport.authz`.

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
from pydantic import BaseModel, Field, conlist

from membrane.auth import AuthContext
from membrane.compute.cpu import CPU
from membrane.serialization import JsonDict
from membrane.transport.authz import enforce_route_scope
from membrane.transport.ops import (
    MAX_BODY_BYTES,
    op_delete,
    op_gossip,
    op_heartbeat,
    op_inventory,
    op_join,
    op_leave,
    op_metrics,
    op_peers,
    op_prefill,
    op_purge,
    op_replicate,
    op_retrieve,
    op_store,
    op_sync,
    op_tombstone,
    op_verify_received,
)

logger = logging.getLogger(__name__)


def _authenticator_for(app: FastAPI) -> object | None:
    """Return the cluster's authenticator from app.state, if any.

    When ``app.state.authenticator`` was populated by the Server at
    startup with an
    :class:`~membrane.auth.mtls.MTLSAuthenticator`, the route
    handler calls :func:`membrane.transport.authz.enforce_route_scope`
    against it. Absent the cluster is treated as non-mTLS
    (single-node deployments).
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


def _scope(request: Request, method: str, path: str) -> AuthContext:
    """Run the per-route scope check.

    Args:
        request: The inbound FastAPI request.
        method: HTTP method.
        path: URL path.

    Returns:
        AuthContext: The authenticated caller's context.
    """
    return enforce_route_scope(
        authenticator=_authenticator_for(request.app),
        method=method,
        path=path,
        headers=_peer_headers(request),
    )


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class FragmentPayload(BaseModel):
    """Wire format for a serialized Fragment.

    Mirrors the 3.0 canonical schema in
    :mod:`membrane.serialization`. The :class:`~membrane.identity.PayloadIdentity`
    is carried as a nested ``identity`` object; ranges are JSON
    arrays of two ints; ``shape`` is a list of ints; ``consistency``
    is one of strong / quorum / eventual; ``hlc`` is the wire
    integer from :class:`~membrane.hlc.HLC`.
    """

    schema_version: int = 5
    tenant_id: str = Field(default="public", max_length=128)
    identity: dict[str, Any] = Field(max_length=64)
    payload_ref: str | None = Field(default=None, max_length=512)
    payload_size: int = Field(ge=0, le=MAX_BODY_BYTES)
    ttl: float
    reuse_score: float
    version_id: int
    consistency: str = "strong"
    hlc: int = 0
    fingerprint_compat: str = Field(default="", max_length=128)

    def to_wire_dict(self) -> JsonDict:
        """Transform into the canonical wire dict accepted by
        :func:`membrane.serialization.from_dict`."""
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "identity": self.identity,
            "payload_ref": self.payload_ref,
            "payload_size": self.payload_size,
            "ttl": self.ttl,
            "reuse_score": self.reuse_score,
            "version_id": self.version_id,
            "consistency": self.consistency,
            "hlc": self.hlc,
            "fingerprint_compat": self.fingerprint_compat,
        }


class StoreRequest(BaseModel):
    """``POST /store`` body."""

    fragment: FragmentPayload
    is_primary: bool = False


class ReplicateRequest(BaseModel):
    """``POST /replicate`` body."""

    fragment: FragmentPayload


class PrefillRequest(BaseModel):
    """``POST /prefill`` body.

    The ``prompt_tokens`` cap is generous (32768) so a long
    agentic prompt still fits; the per-token range is restricted
    to a valid int32 so a hostile payload cannot smuggle
    float / NaN values into the wire.
    """

    prompt_tokens: conlist(int, max_length=32768)  # type: ignore[valid-type]
    model_id: str = Field(default="default", max_length=256)


class SyncRequest(BaseModel):
    """``POST /sync`` body."""

    source_url: str = Field(max_length=2048)


class JoinRequest(BaseModel):
    """``POST /join`` body."""

    node_id: str = Field(min_length=1, max_length=128)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(ge=1, le=65535)


class LeaveRequest(BaseModel):
    """``POST /leave`` body."""

    node_id: str = Field(min_length=1, max_length=128)


class GossipRequest(BaseModel):
    """``POST /gossip`` body.

    The peers + fragment_locations lists are capped so a
    malicious peer cannot blow out the gossip budget with a
    million-element payload.
    """

    peers: list[dict[str, Any]] = Field(default_factory=list, max_length=4096)
    fragment_locations: dict[str, list[str]] = Field(
        default_factory=dict, max_length=131072
    )
    inventory_digest: dict[str, int] = Field(default_factory=dict, max_length=131072)


class DeleteRequest(BaseModel):
    """``POST /delete`` body."""

    content_hash: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=128)
    tombstone_until: float | None = None


class TombstoneRequest(BaseModel):
    """``POST /tombstone`` body."""

    content_hash: str = Field(min_length=1, max_length=128)
    until: float
    node_id: str = Field(min_length=1, max_length=128)


class PurgeRequest(BaseModel):
    """``POST /purge`` body."""

    content_hash: str = Field(min_length=1, max_length=128)


class VerifyRequest(BaseModel):
    """``POST /verify`` body."""

    content_hash: str = Field(min_length=1, max_length=128)
    claimed_size: int = Field(ge=0, le=MAX_BODY_BYTES)
    claimed_sha256_hex: str = Field(min_length=64, max_length=64)


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


def livez(_app: FastAPI) -> dict[str, str]:
    """``GET /livez`` — liveness probe (public)."""
    return {"status": "alive"}


def readyz(app: FastAPI):
    """``GET /readyz`` — readiness probe (public, deep)."""
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
    :mod:`membrane.transport.ops` after running the per-route
    scope check in :mod:`membrane.transport.authz`. The Pydantic
    models at the top of this module validate request bodies; the
    inline lambdas capture ``app`` so the registered functions
    take only the request body.

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
    def store_handler(req: StoreRequest, request: Request):
        return _store(app, req, request)

    def replicate_handler(req: ReplicateRequest, request: Request):
        return _replicate(app, req, request)

    def sync_handler(req: SyncRequest, request: Request):
        return _sync(app, req, request)

    def prefill_handler(req: PrefillRequest, request: Request):
        return _prefill(app, req, request)

    def join_handler(req: JoinRequest, request: Request):
        return _join(app, req, request)

    def leave_handler(req: LeaveRequest, request: Request):
        return _leave(app, req, request)

    def gossip_handler(req: GossipRequest, request: Request):
        return _gossip(app, req, request)

    def delete_handler(req: DeleteRequest, request: Request):
        return _delete(app, req, request)

    def tombstone_handler(req: TombstoneRequest, request: Request):
        return _tombstone(app, req, request)

    def purge_handler(req: PurgeRequest, request: Request):
        return _purge(app, req, request)

    def verify_handler(req: VerifyRequest, request: Request):
        return _verify(app, req, request)

    app.add_api_route("/store", store_handler, methods=["POST"], response_model=None)
    app.add_api_route("/replicate", replicate_handler, methods=["POST"], response_model=None)
    app.add_api_route("/sync", sync_handler, methods=["POST"], response_model=None)
    app.add_api_route("/prefill", prefill_handler, methods=["POST"], response_model=None)
    app.add_api_route("/join", join_handler, methods=["POST"], response_model=None)
    app.add_api_route("/leave", leave_handler, methods=["POST"], response_model=None)
    app.add_api_route("/gossip", gossip_handler, methods=["POST"], response_model=None)
    app.add_api_route("/delete", delete_handler, methods=["POST"], response_model=None)
    app.add_api_route("/tombstone", tombstone_handler, methods=["POST"], response_model=None)
    app.add_api_route("/purge", purge_handler, methods=["POST"], response_model=None)
    app.add_api_route("/verify", verify_handler, methods=["POST"], response_model=None)


def _heartbeat(app: FastAPI, request: Request):
    context = _scope(request, "GET", "/heartbeat")
    status, body = op_heartbeat(
        app.state.node,
        cluster=getattr(app.state, "cluster_manager", None),
        headers=_peer_headers(request),
        auth_context=context,
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


def _store(app: FastAPI, req: StoreRequest, request: Request):
    context = _scope(request, "POST", "/store")
    status, body = op_store(
        app.state.node,
        req.fragment.to_wire_dict(),
        req.is_primary,
        cluster=getattr(app.state, "cluster_manager", None),
        quorum_attempt=getattr(app.state, "quorum_attempt", None),
        draining=bool(getattr(app.state.server, "is_draining", False))
        if hasattr(app.state, "server")
        else False,
        auth_context=context,
    )
    return _respond(status, body)


def _respond_with_retry_after(status: int, body: JsonDict, retry_after: int) -> Response:
    return JSONResponse(
        body,
        status_code=status,
        headers={"Retry-After": str(retry_after)},
    )


def _replicate(app: FastAPI, req: ReplicateRequest, request: Request):
    context = _scope(request, "POST", "/replicate")
    status, body = op_replicate(
        app.state.node,
        req.fragment.to_wire_dict(),
        auth_context=context,
    )
    return _respond(status, body)


def _sync(app: FastAPI, req: SyncRequest, request: Request):
    context = _scope(request, "POST", "/sync")
    status, body = op_sync(
        app.state.node,
        app.state.transfer_service,
        req.source_url,
        auth_context=context,
    )
    return _respond(status, body)


def _prefill(app: FastAPI, req: PrefillRequest, request: Request):
    context = _scope(request, "POST", "/prefill")
    status, body = op_prefill(
        app.state.node,
        app.state.compute_backend or CPU(),
        req.prompt_tokens,
        req.model_id,
        auth_context=context,
    )
    return _respond(status, body)


def _join(app: FastAPI, req: JoinRequest, request: Request):
    context = _scope(request, "POST", "/join")
    status, body = op_join(
        app.state.cluster_manager,
        req.node_id,
        req.host,
        req.port,
        headers=_peer_headers(request),
        authenticator=_authenticator_for(app),
        auth_context=context,
    )
    return _respond(status, body)


def _leave(app: FastAPI, req: LeaveRequest, request: Request):
    context = _scope(request, "POST", "/leave")
    status, body = op_leave(
        app.state.cluster_manager,
        req.node_id,
        auth_context=context,
    )
    return _respond(status, body)


def _gossip(app: FastAPI, req: GossipRequest, request: Request):
    context = _scope(request, "POST", "/gossip")
    status, body = op_gossip(
        app.state.cluster_manager,
        req.model_dump(),
        auth_context=context,
    )
    return _respond(status, body)


def _delete(app: FastAPI, req: DeleteRequest, request: Request):
    context = _scope(request, "POST", "/delete")
    status, body = op_delete(
        app.state.node,
        getattr(app.state, "tombstones", None),
        req.content_hash,
        req.node_id,
        req.tombstone_until,
        auth_context=context,
    )
    return _respond(status, body)


def _tombstone(app: FastAPI, req: TombstoneRequest, request: Request):
    context = _scope(request, "POST", "/tombstone")
    status, body = op_tombstone(
        getattr(app.state, "tombstones", None),
        req.content_hash,
        req.until,
        req.node_id,
        auth_context=context,
    )
    return _respond(status, body)


def _purge(app: FastAPI, req: PurgeRequest, request: Request):
    context = _scope(request, "POST", "/purge")
    status, body = op_purge(
        app.state.node,
        getattr(app.state, "tombstones", None),
        req.content_hash,
        auth_context=context,
    )
    return _respond(status, body)


def _verify(app: FastAPI, req: VerifyRequest, request: Request):
    context = _scope(request, "POST", "/verify")
    status, body = op_verify_received(
        app.state.node,
        req.content_hash,
        req.claimed_size,
        req.claimed_sha256_hex,
        auth_context=context,
    )
    return _respond(status, body)


def _respond(status: int, body: JsonDict | tuple[str, dict[str, str]] | object) -> Response:
    """Translate an operation's ``(status, body)`` to a FastAPI response."""
    if status == 200:
        return JSONResponse(body)
    return JSONResponse(body, status_code=status)


__all__ = ["MAX_BODY_BYTES", "register_routes"]
