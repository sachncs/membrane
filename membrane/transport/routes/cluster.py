"""Peer routes: replication, membership, gossip, and the peer list."""

from fastapi import FastAPI, Request

from membrane.security.tenant import has_admin_scope
from membrane.transport.context import app_context
from membrane.transport.metrics import record_transport
from membrane.transport.ops import (
    op_gossip,
    op_join,
    op_leave,
    op_peers,
    op_replicate,
)
from membrane.transport.routes.common import (
    authenticator_for,
    peer_headers,
    respond,
    route_scope,
    transport_metrics_for,
)
from membrane.transport.routes.models import (
    GossipRequest,
    JoinRequest,
    LeaveRequest,
    ReplicateRequest,
)


def handle_replicate(app: FastAPI, req: ReplicateRequest, request: Request):
    """Serve ``POST /replicate`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/replicate")
    status, body = record_transport(
        transport_metrics_for(app),
        "replicate",
        "POST",
        lambda: op_replicate(
            app_context(app).node,
            req.fragment.to_wire_dict(),
            auth_context=context,
            # Only peers (admin) may hand over primary ownership.
            # (An empty subject means authentication is off.)
            is_primary=req.is_primary and (not context.subject or has_admin_scope(context.scopes)),
        ),
    )
    return respond(status, body)


def handle_join(app: FastAPI, req: JoinRequest, request: Request):
    """Serve ``POST /join`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/join")
    status, body = record_transport(
        transport_metrics_for(app),
        "join",
        "POST",
        lambda: op_join(
            app_context(app).cluster,
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
    """Serve ``POST /leave`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/leave")
    status, body = record_transport(
        transport_metrics_for(app),
        "leave",
        "POST",
        lambda: op_leave(
            app_context(app).cluster,
            req.node_id,
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_gossip(app: FastAPI, req: GossipRequest, request: Request):
    """Serve ``POST /gossip`` after the scope check.

    Args:
        app: The FastAPI application.
        req: Validated request body.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "POST", "/gossip")
    status, body = record_transport(
        transport_metrics_for(app),
        "gossip",
        "POST",
        lambda: op_gossip(
            app_context(app).cluster,
            req.model_dump(),
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_peers(app: FastAPI):
    """Serve ``GET /peers``.

    Args:
        app: The FastAPI application.

    Returns:
        object: The HTTP response.
    """
    status, body = record_transport(
        transport_metrics_for(app),
        "peers",
        "GET",
        lambda: op_peers(app_context(app).cluster),
    )
    return respond(status, body)


__all__ = ["handle_gossip", "handle_join", "handle_leave", "handle_peers", "handle_replicate"]
