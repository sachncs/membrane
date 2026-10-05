"""Delete, tombstone, purge, and verify routes (admin scope)."""

from fastapi import FastAPI, Request

from membrane.transport.context import app_context
from membrane.transport.metrics import record_transport
from membrane.transport.ops import (
    op_delete,
    op_purge,
    op_tombstone,
    op_verify_received,
)
from membrane.transport.routes.common import (
    respond,
    route_scope,
    transport_metrics_for,
)
from membrane.transport.routes.models import (
    DeleteRequest,
    PurgeRequest,
    TombstoneRequest,
    VerifyRequest,
)


def handle_delete(app: FastAPI, req: DeleteRequest, request: Request):
    """Serve ``POST /delete`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/delete")
    status, body = record_transport(
        transport_metrics_for(app),
        "delete",
        "POST",
        lambda: op_delete(
            app_context(app).node,
            app_context(app).tombstones,
            req.content_hash,
            req.node_id,
            req.tombstone_until,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_tombstone(app: FastAPI, req: TombstoneRequest, request: Request):
    """Serve ``POST /tombstone`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/tombstone")
    status, body = record_transport(
        transport_metrics_for(app),
        "tombstone",
        "POST",
        lambda: op_tombstone(
            app_context(app).tombstones,
            req.content_hash,
            req.until,
            req.node_id,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_purge(app: FastAPI, req: PurgeRequest, request: Request):
    """Serve ``POST /purge`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/purge")
    status, body = record_transport(
        transport_metrics_for(app),
        "purge",
        "POST",
        lambda: op_purge(
            app_context(app).node,
            app_context(app).tombstones,
            req.content_hash,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_verify(app: FastAPI, req: VerifyRequest, request: Request):
    """Serve ``POST /verify`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/verify")
    status, body = record_transport(
        transport_metrics_for(app),
        "verify",
        "POST",
        lambda: op_verify_received(
            app_context(app).node,
            req.content_hash,
            req.claimed_size,
            req.claimed_sha256_hex,
            auth_context=context,
        ),
    )
    return respond(status, body)


__all__ = ["handle_delete", "handle_purge", "handle_tombstone", "handle_verify"]
