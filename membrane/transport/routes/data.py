"""Fragment, blob, prefill, and sync routes."""

import asyncio

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from membrane.auth import AuthContext
from membrane.codec import COMPRESSION_METHODS, CompressionTransport
from membrane.compute.cpu import CPU
from membrane.integrity import record_corrupt_payload
from membrane.metrics import NodeMetrics
from membrane.network.peer import ACCEPT_COMPRESSION_HEADER, COMPRESS_MIN_BYTES, COMPRESSION_HEADER
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
    valid_payload_ref,
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
from membrane.transport.uploads import MAX_CHUNK_BYTES, UploadError, UploadRegistry
from membrane.wire.v3.chunks import sha256_hex


def handle_retrieve(app: FastAPI, content_hash: str, context: AuthContext, session_id: str = ""):
    """Serve ``GET /retrieve`` for the authenticated caller.

    A hit is recorded (reuse score, promotion demand, and the session's
    history). On a regional cache, a miss is fetched from the origin.

    Args:
        app: The FastAPI application.
        content_hash: Content hash of the fragment.
        context: Authenticated caller.
        session_id: The ``X-Membrane-Session`` header, if sent.

    Returns:
        object: The HTTP response.
    """
    services = app_context(app).services

    def retrieve() -> tuple[int, object]:
        node = app_context(app).node
        status, body = op_retrieve(node, content_hash, auth_context=context)
        found = isinstance(body, dict) and body.get("found")
        origin = services.origin if services is not None else None
        if status == 200 and not found and origin is not None and origin.fetch(content_hash):
            status, body = op_retrieve(node, content_hash, auth_context=context)
        return status, body

    status, body = record_transport(transport_metrics_for(app), "retrieve", "GET", retrieve)
    registry = app_context(app).metrics_registry
    hit = status == 200 and isinstance(body, dict) and bool(body.get("found"))
    if hit and services is not None:
        services.memory.record_access(content_hash, session_id[:256], context.subject)
    if registry is not None and status == 200 and isinstance(body, dict):
        result = "hit" if hit else "corrupt" if body.get("corrupt") else "miss"
        NodeMetrics(registry).cache_lookups.inc(result=result)
        if result == "corrupt":
            record_corrupt_payload(registry)
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
    services = app_context(app).services
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
            store_guard=services.memory.check_store if services is not None else None,
        ),
    )
    return respond(status, body)


def handle_inventory(app: FastAPI, after: str = "", limit: int = 0):
    """Serve ``GET /inventory`` (``?after=&limit=`` pages through large nodes).

    Args:
        app: The FastAPI application.
        after: Page cursor.
        limit: Page size; ``0`` for everything (capped at 100,000 per page).

    Returns:
        object: The HTTP response.
    """
    limit = min(max(limit, 0), 100_000)
    status, body = record_transport(
        transport_metrics_for(app),
        "inventory",
        "GET",
        lambda: op_inventory(app_context(app).node, after=after, limit=limit),
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
    services = app_context(app).services
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
            stamp=services.memory.stamp if services is not None else None,
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
    if request.headers.get(COMPRESSION_HEADER.lower()):
        try:
            data = await asyncio.to_thread(CompressionTransport().decompress, data, MAX_BODY_BYTES)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"error": "bad compressed body", "detail": str(exc)}, status_code=400)
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
    headers = {"X-Content-SHA256": sha256_hex(data)}
    if request.method == "HEAD":
        headers["Content-Length"] = str(len(data))
        return Response(status_code=200, headers=headers, media_type="application/octet-stream")
    accepted = request.headers.get(ACCEPT_COMPRESSION_HEADER.lower(), "raw")
    if accepted in COMPRESSION_METHODS and accepted != "raw" and len(data) >= COMPRESS_MIN_BYTES:
        data = await asyncio.to_thread(CompressionTransport(accepted).compress, data)
        headers[COMPRESSION_HEADER] = accepted
    return Response(content=data, headers=headers, media_type="application/octet-stream")


async def handle_begin_upload(app: FastAPI, payload_ref: str, request: Request) -> Response:
    """Serve ``POST /blobs/{payload_ref}/upload``: start or resume a chunked upload.

    Args:
        app: The FastAPI application.
        payload_ref: Content-store key.
        request: Body ``{"chunk_size", "total_bytes", "chunks": [sha256...], "sha256"}``.

    Returns:
        Response: ``{"received": [indices]}``, ``{"stored": true}`` when the
        bytes are already here, or ``400`` / ``429``.
    """
    route_scope(request, "POST", "/blobs/upload")
    node = app_context(app).node
    if node is None or not valid_payload_ref(payload_ref):
        return JSONResponse({"error": "invalid payload_ref"}, status_code=400)
    if node.content_store.has(payload_ref):
        return JSONResponse({"stored": True, "received": []})
    body = await request.json()
    try:
        received = uploads_for(app).begin(
            payload_ref,
            int(body["chunk_size"]),
            int(body["total_bytes"]),
            [str(c) for c in body["chunks"]],
            str(body["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        status = 429 if "retry later" in str(exc) else 400
        return JSONResponse({"error": str(exc)}, status_code=status)
    return JSONResponse({"stored": False, "received": received})


async def handle_upload_chunk(app: FastAPI, payload_ref: str, index: int, request: Request) -> Response:
    """Serve ``PUT /blobs/{payload_ref}/upload/{index}``: one chunk of an upload.

    Args:
        app: The FastAPI application.
        payload_ref: Content-store key.
        index: Chunk index.
        request: The chunk bytes (optionally compressed).

    Returns:
        Response: ``{"stored": true}`` once the last chunk completes the
        verified payload, else ``{"stored": false, "received": [...]}``.
    """
    route_scope(request, "PUT", "/blobs/upload")
    data = await read_limited_body(request, MAX_CHUNK_BYTES + 64)
    if data is None:
        return JSONResponse({"error": "chunk too large"}, status_code=413)
    if request.headers.get(COMPRESSION_HEADER.lower()):
        try:
            data = CompressionTransport().decompress(data, MAX_CHUNK_BYTES)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"error": "bad compressed body", "detail": str(exc)}, status_code=400)
    registry = uploads_for(app)
    try:
        payload = await asyncio.to_thread(registry.add_chunk, payload_ref, index, data)
    except UploadError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if payload is None:
        return JSONResponse({"stored": False, "received": registry.received(payload_ref)})
    node = app_context(app).node
    if node is None:
        return JSONResponse({"error": "no node"}, status_code=503)
    await asyncio.to_thread(node.content_store.put, payload_ref, payload)
    return JSONResponse({"stored": True, "received": []})


def uploads_for(app: FastAPI) -> UploadRegistry:
    """Return the app's upload registry, creating it on first use.

    Args:
        app: The FastAPI application.

    Returns:
        UploadRegistry: Staged uploads.
    """
    registry = getattr(app.state, "uploads", None)
    if registry is None:
        registry = UploadRegistry(max_bytes=MAX_BODY_BYTES)
        app.state.uploads = registry
    return registry


__all__ = [
    "handle_begin_upload",
    "handle_get_blob",
    "handle_inventory",
    "handle_prefill",
    "handle_put_blob",
    "handle_retrieve",
    "handle_store",
    "handle_sync",
    "handle_upload_chunk",
    "uploads_for",
]
