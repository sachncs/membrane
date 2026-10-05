"""FastAPI route bindings.

This module binds the operations in :mod:`membrane.transport.ops`
to FastAPI endpoints. The Pydantic models at the top describe the
request bodies; the handlers delegate to the corresponding
operation after running the per-route scope check from
:mod:`membrane.transport.authz`.

Every route except the ``/livez`` and ``/readyz`` probes runs the
scope check, so a node configured with an authenticator never
serves an unauthenticated read.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, conlist

from membrane.auth import AuthBackendError, AuthContext, AuthForbiddenError
from membrane.compute.cpu import CPU
from membrane.metrics import TransportMetrics
from membrane.serialization import JsonDict
from membrane.transport.authz import enforce_route_scope
from membrane.transport.metrics import record_transport
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
from membrane.transport.tls_protocol import peer_headers_from_scope

logger = logging.getLogger(__name__)


def authenticator_for(app: FastAPI) -> object | None:
    """Return the cluster's authenticator from app.state, if any.

    When ``app.state.authenticator`` was populated by the Server at
    startup with an
    :class:`~membrane.auth.mtls.MTLSAuthenticator`, the route
    handler calls :func:`membrane.transport.authz.enforce_route_scope`
    against it. Absent the cluster is treated as non-mTLS
    (single-node deployments).
    """
    return getattr(app.state, "authenticator", None)


def peer_headers(request: Request) -> dict[str, str]:
    """Capture inbound headers as a plain dict for the authenticator.

    FastAPI's :class:`Request.headers` is case-insensitive; we
    downcase keys here so
    :meth:`membrane.transport.tls.parse_peer_cn_header` reads them
    with the ``x-ssl-client-cn`` spelling it expects.

    A client-supplied ``x-ssl-client-cn`` header is always
    discarded. The only trusted source is the CN of the verified
    peer certificate, which :class:`~membrane.transport.tls_protocol.PeerCertH11Protocol`
    records in the ASGI scope after the TLS handshake.
    """
    return peer_headers_from_scope(request.scope, request.headers.items())


def route_scope(request: Request, method: str, path: str) -> AuthContext:
    """Run the per-route scope check.

    Args:
        request: The inbound FastAPI request.
        method: HTTP method.
        path: URL path.

    Returns:
        AuthContext: The authenticated caller's context.
    """
    return enforce_route_scope(
        authenticator=authenticator_for(request.app),
        method=method,
        path=path,
        headers=peer_headers(request),
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

    The fields mirror :meth:`membrane.network.gossip.GossipState.to_json`;
    any field dropped here never reaches the receiver.
    """

    node_id: str = Field(min_length=1, max_length=128)
    timestamp: float
    peers: list[dict[str, Any]] = Field(default_factory=list, max_length=4096)
    fragment_locations: dict[str, list[str]] = Field(default_factory=dict, max_length=131072)
    inventory_bloom: str = Field(default="", max_length=4 * 1024 * 1024)
    inventory_merkle_root: str = Field(default="", max_length=128)
    inventory_size: int = Field(default=0, ge=0)
    fragment_tombstones: dict[str, float] = Field(default_factory=dict, max_length=131072)
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
    # A full node is healthy: ``Node.store`` evicts to make room, so
    # memory saturation is the steady state of a warm cache and must
    # not take the pod out of rotation.
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def auth_error_response(_request: Request, exc: Exception) -> Response:
    """Translate authentication / authorization failures to 401 / 403.

    The body is deliberately generic so a probe cannot distinguish
    an unknown key from a known key with the wrong scope beyond the
    status code itself.
    """
    if isinstance(exc, AuthForbiddenError):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return JSONResponse(
        {"error": "unauthorized"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


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
    app.add_exception_handler(AuthBackendError, auth_error_response)

    # GET endpoints that have no body.
    def heartbeat_handler(request: Request):
        return handle_heartbeat(app, request)

    app.add_api_route("/heartbeat", heartbeat_handler, methods=["GET"])
    app.add_api_route("/livez", lambda: livez(app), methods=["GET"])
    app.add_api_route("/readyz", lambda: readyz(app), methods=["GET"])

    def metrics_handler(request: Request):
        route_scope(request, "GET", "/metrics")
        return handle_metrics(app)

    def metrics_json_handler(request: Request):
        route_scope(request, "GET", "/metrics.json")
        return metrics_json(app)

    def retrieve_handler(content_hash: str, request: Request):
        return handle_retrieve(app, content_hash, route_scope(request, "GET", "/retrieve"))

    def inventory_handler(request: Request):
        route_scope(request, "GET", "/inventory")
        return handle_inventory(app)

    def peers_handler(request: Request):
        route_scope(request, "GET", "/peers")
        return handle_peers(app)

    app.add_api_route("/metrics", metrics_handler, methods=["GET"])
    app.add_api_route("/metrics.json", metrics_json_handler, methods=["GET"])
    app.add_api_route("/retrieve", retrieve_handler, methods=["GET"])
    app.add_api_route("/inventory", inventory_handler, methods=["GET"])
    app.add_api_route("/peers", peers_handler, methods=["GET"])

    # POST endpoints.
    def store_handler(req: StoreRequest, request: Request):
        return handle_store(app, req, request)

    def replicate_handler(req: ReplicateRequest, request: Request):
        return handle_replicate(app, req, request)

    def sync_handler(req: SyncRequest, request: Request):
        return handle_sync(app, req, request)

    def prefill_handler(req: PrefillRequest, request: Request):
        return handle_prefill(app, req, request)

    def join_handler(req: JoinRequest, request: Request):
        return handle_join(app, req, request)

    def leave_handler(req: LeaveRequest, request: Request):
        return handle_leave(app, req, request)

    def gossip_handler(req: GossipRequest, request: Request):
        return handle_gossip(app, req, request)

    def delete_handler(req: DeleteRequest, request: Request):
        return handle_delete(app, req, request)

    def tombstone_handler(req: TombstoneRequest, request: Request):
        return handle_tombstone(app, req, request)

    def purge_handler(req: PurgeRequest, request: Request):
        return handle_purge(app, req, request)

    def verify_handler(req: VerifyRequest, request: Request):
        return handle_verify(app, req, request)

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


def transport_metrics_for(app: FastAPI) -> TransportMetrics | None:
    """Return the cluster's TransportMetrics from app.state, if any.

    The :class:`Server` populates ``app.state.transport_metrics``
    at construction; transport handlers use the helper to
    record every op invocation.
    """
    return getattr(app.state, "transport_metrics", None)


def handle_heartbeat(app: FastAPI, request: Request):
    context = route_scope(request, "GET", "/heartbeat")
    status, body = record_transport(
        transport_metrics_for(app),
        "heartbeat",
        "GET",
        lambda: op_heartbeat(
            app.state.node,
            cluster=getattr(app.state, "cluster_manager", None),
            headers=peer_headers(request),
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_metrics(app: FastAPI):
    """``GET /metrics`` — Prometheus text or legacy JSON."""
    refresh = getattr(app.state, "refresh_metrics", None)
    if refresh is not None:
        refresh()
    status, payload = op_metrics(app.state.node, app.state.metrics_registry)
    if status == 200 and isinstance(payload, tuple):
        text, headers = payload
        return PlainTextResponse(text, media_type=headers["media_type"])
    return respond(status, payload)


def metrics_json(app: FastAPI):
    """``GET /metrics.json`` — legacy JSON snapshot for the TUI."""
    status, body = op_metrics(app.state.node)
    return respond(status, body)


def handle_inventory(app: FastAPI):
    status, body = record_transport(
        transport_metrics_for(app),
        "inventory",
        "GET",
        lambda: op_inventory(app.state.node),
    )
    return respond(status, body)


def handle_peers(app: FastAPI):
    status, body = record_transport(
        transport_metrics_for(app),
        "peers",
        "GET",
        lambda: op_peers(app.state.cluster_manager),
    )
    return respond(status, body)


def handle_retrieve(app: FastAPI, content_hash: str, context: AuthContext):
    status, body = record_transport(
        transport_metrics_for(app),
        "retrieve",
        "GET",
        lambda: op_retrieve(app.state.node, content_hash, auth_context=context),
    )
    return respond(status, body)


def handle_store(app: FastAPI, req: StoreRequest, request: Request):
    context = route_scope(request, "POST", "/store")
    cluster_metrics_obj = getattr(app.state, "cluster_metrics", None)
    status, body = record_transport(
        transport_metrics_for(app),
        "store",
        "POST",
        lambda: op_store(
            app.state.node,
            req.fragment.to_wire_dict(),
            req.is_primary,
            cluster=getattr(app.state, "cluster_manager", None),
            quorum_attempt=getattr(app.state, "quorum_attempt", None),
            draining=bool(getattr(app.state.server, "is_draining", False)) if hasattr(app.state, "server") else False,
            auth_context=context,
            cluster_metrics=cluster_metrics_obj,
        ),
    )
    return respond(status, body)


def respond_with_retry_after(status: int, body: JsonDict, retry_after: int) -> Response:
    return JSONResponse(
        body,
        status_code=status,
        headers={"Retry-After": str(retry_after)},
    )


def handle_replicate(app: FastAPI, req: ReplicateRequest, request: Request):
    context = route_scope(request, "POST", "/replicate")
    status, body = record_transport(
        transport_metrics_for(app),
        "replicate",
        "POST",
        lambda: op_replicate(
            app.state.node,
            req.fragment.to_wire_dict(),
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_sync(app: FastAPI, req: SyncRequest, request: Request):
    context = route_scope(request, "POST", "/sync")
    status, body = record_transport(
        transport_metrics_for(app),
        "sync",
        "POST",
        lambda: op_sync(
            app.state.node,
            app.state.transfer_service,
            req.source_url,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_prefill(app: FastAPI, req: PrefillRequest, request: Request):
    context = route_scope(request, "POST", "/prefill")
    status, body = record_transport(
        transport_metrics_for(app),
        "prefill",
        "POST",
        lambda: op_prefill(
            app.state.node,
            app.state.compute_backend or CPU(),
            req.prompt_tokens,
            req.model_id,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_join(app: FastAPI, req: JoinRequest, request: Request):
    context = route_scope(request, "POST", "/join")
    status, body = record_transport(
        transport_metrics_for(app),
        "join",
        "POST",
        lambda: op_join(
            app.state.cluster_manager,
            req.node_id,
            req.host,
            req.port,
            headers=peer_headers(request),
            authenticator=authenticator_for(app),
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_leave(app: FastAPI, req: LeaveRequest, request: Request):
    context = route_scope(request, "POST", "/leave")
    status, body = record_transport(
        transport_metrics_for(app),
        "leave",
        "POST",
        lambda: op_leave(
            app.state.cluster_manager,
            req.node_id,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_gossip(app: FastAPI, req: GossipRequest, request: Request):
    context = route_scope(request, "POST", "/gossip")
    status, body = record_transport(
        transport_metrics_for(app),
        "gossip",
        "POST",
        lambda: op_gossip(
            app.state.cluster_manager,
            req.model_dump(),
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_delete(app: FastAPI, req: DeleteRequest, request: Request):
    context = route_scope(request, "POST", "/delete")
    status, body = record_transport(
        transport_metrics_for(app),
        "delete",
        "POST",
        lambda: op_delete(
            app.state.node,
            getattr(app.state, "tombstones", None),
            req.content_hash,
            req.node_id,
            req.tombstone_until,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_tombstone(app: FastAPI, req: TombstoneRequest, request: Request):
    context = route_scope(request, "POST", "/tombstone")
    status, body = record_transport(
        transport_metrics_for(app),
        "tombstone",
        "POST",
        lambda: op_tombstone(
            getattr(app.state, "tombstones", None),
            req.content_hash,
            req.until,
            req.node_id,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_purge(app: FastAPI, req: PurgeRequest, request: Request):
    context = route_scope(request, "POST", "/purge")
    status, body = record_transport(
        transport_metrics_for(app),
        "purge",
        "POST",
        lambda: op_purge(
            app.state.node,
            getattr(app.state, "tombstones", None),
            req.content_hash,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_verify(app: FastAPI, req: VerifyRequest, request: Request):
    context = route_scope(request, "POST", "/verify")
    status, body = record_transport(
        transport_metrics_for(app),
        "verify",
        "POST",
        lambda: op_verify_received(
            app.state.node,
            req.content_hash,
            req.claimed_size,
            req.claimed_sha256_hex,
            auth_context=context,
        ),
    )
    return respond(status, body)


def respond(status: int, body: JsonDict | tuple[str, dict[str, str]] | object) -> Response:
    """Translate an operation's ``(status, body)`` to a FastAPI response."""
    if status == 200:
        return JSONResponse(body)
    return JSONResponse(body, status_code=status)


__all__ = ["MAX_BODY_BYTES", "register_routes"]
