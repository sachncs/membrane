"""Fragment, blob, prefill, and sync routes."""

import asyncio

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from membrane.auth import AuthContext
from membrane.compute.cpu import CPU
from membrane.transport.context import app_context
from membrane.transport.metrics import record_transport
from membrane.transport.ops import (
    MAX_BODY_BYTES,
    op_get_blob,
    op_inventory,
    op_prefill,
    op_put_blob,
    op_retrieve,
    op_store,
    op_sync,
)
from membrane.transport.routes.common import (
    read_limited_body,
    respond,
    route_scope,
    transport_metrics_for,
)
from membrane.transport.routes.models import (
    PrefillRequest,
    StoreRequest,
    SyncRequest,
)
from membrane.wire.v3.chunks import sha256_hex


def handle_retrieve(app: FastAPI, content_hash: str, context: AuthContext):
    """Serve ``GET /retrieve`` for the authenticated caller.

    Args:
        app: The FastAPI application.
        content_hash: Content hash of the fragment.
        context: Authenticated caller.

    Returns:
        object: The HTTP response.
    """
    status, body = record_transport(
        transport_metrics_for(app),
        "retrieve",
        "GET",
        lambda: op_retrieve(app_context(app).node, content_hash, auth_context=context),
    )
    return respond(status, body)


def handle_store(app: FastAPI, req: StoreRequest, request: Request):
    """Serve ``POST /store`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/store")
    cluster_metrics_obj = app_context(app).cluster_metrics
    status, body = record_transport(
        transport_metrics_for(app),
        "store",
        "POST",
        lambda: op_store(
            app_context(app).node,
            req.fragment.to_wire_dict(),
            req.is_primary,
            cluster=app_context(app).cluster,
            quorum_attempt=app_context(app).quorum_attempt,
            draining=app_context(app).draining,
            auth_context=context,
            cluster_metrics=cluster_metrics_obj,
        ),
    )
    return respond(status, body)


def handle_inventory(app: FastAPI):
    """Serve ``GET /inventory``.

    Args:
        app: The FastAPI application.

    Returns:
        object: The HTTP response.
    """
    status, body = record_transport(
        transport_metrics_for(app),
        "inventory",
        "GET",
        lambda: op_inventory(app_context(app).node),
    )
    return respond(status, body)


def handle_prefill(app: FastAPI, req: PrefillRequest, request: Request):
    """Serve ``POST /prefill`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/prefill")
    status, body = record_transport(
        transport_metrics_for(app),
        "prefill",
        "POST",
        lambda: op_prefill(
            app_context(app).node,
            app_context(app).compute_backend or CPU(),
            req.prompt_tokens,
            req.model_id,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_sync(app: FastAPI, req: SyncRequest, request: Request):
    """Serve ``POST /sync`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/sync")
    status, body = record_transport(
        transport_metrics_for(app),
        "sync",
        "POST",
        lambda: op_sync(
            app_context(app).node,
            app_context(app).transfer_service,
            req.source_url,
            auth_context=context,
        ),
    )
    return respond(status, body)


async def handle_put_blob(app: FastAPI, payload_ref: str, request: Request) -> Response:
    """Serve ``PUT /blobs/{payload_ref}``: store KV bytes sent by a peer.

    Args:
        app: The FastAPI application.
        payload_ref: Content-store key from the path.
        request: The inbound request; the body is the raw bytes and
            ``X-Content-SHA256`` their digest.

    Returns:
        Response: ``200``, ``400`` (bad key or digest), or ``413`` (too large).
    """
    route_scope(request, "PUT", "/blobs")
    data = await read_limited_body(request, MAX_BODY_BYTES)
    if data is None:
        return JSONResponse({"error": "payload too large", "limit": MAX_BODY_BYTES}, status_code=413)
    claimed = request.headers.get("x-content-sha256", "")
    status, body = await asyncio.to_thread(
        record_transport,
        transport_metrics_for(app),
        "blobs",
        "PUT",
        lambda: op_put_blob(app_context(app).node, payload_ref, data, claimed),
    )
    return respond(status, body)


async def handle_get_blob(app: FastAPI, payload_ref: str, request: Request) -> Response:
    """Serve ``GET`` / ``HEAD /blobs/{payload_ref}``: KV bytes for a peer.

    Args:
        app: The FastAPI application.
        payload_ref: Content-store key from the path.
        request: The inbound request.

    Returns:
        Response: The bytes with ``X-Content-SHA256`` (``HEAD``: headers
        only), ``404`` when absent, or ``400`` for a bad key.
    """
    route_scope(request, request.method, "/blobs")
    status, data = await asyncio.to_thread(op_get_blob, app_context(app).node, payload_ref)
    if data is None:
        return JSONResponse({"error": "not found" if status == 404 else "invalid payload_ref"}, status_code=status)
    headers = {"X-Content-SHA256": sha256_hex(data), "Content-Length": str(len(data))}
    if request.method == "HEAD":
        return Response(status_code=200, headers=headers, media_type="application/octet-stream")
    return Response(content=data, headers=headers, media_type="application/octet-stream")


__all__ = [
    "handle_get_blob",
    "handle_inventory",
    "handle_prefill",
    "handle_put_blob",
    "handle_retrieve",
    "handle_store",
    "handle_sync",
]
