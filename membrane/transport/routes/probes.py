"""Health probes, heartbeat, and metrics routes."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from membrane.transport.context import app_context
from membrane.transport.metrics import record_transport
from membrane.transport.ops import (
    op_heartbeat,
    op_metrics,
)
from membrane.transport.routes.common import (
    peer_headers,
    respond,
    route_scope,
    transport_metrics_for,
)


def livez(_app: FastAPI) -> dict[str, str]:
    """``GET /livez`` — liveness probe (public).

    Returns:
        dict[str, str]: ``{"status": "alive"}``.
    """
    return {"status": "alive"}


def readyz(app: FastAPI):
    """``GET /readyz`` — readiness probe (public, deep).

    Args:
        app: The FastAPI application.

    Returns:
        object: ``{"status": "ready"}``, or a 503 response without a node
        or while the node drains (so load balancers stop routing to it).
    """
    if not app_context(app).node:
        return JSONResponse({"status": "no node"}, status_code=503)
    server = app_context(app).server
    if server is not None and getattr(server, "is_draining", False):
        return JSONResponse({"status": "draining"}, status_code=503, headers={"Retry-After": "5"})
    # A full node is healthy: ``Node.store`` evicts to make room, so
    # memory saturation is the steady state of a warm cache and must
    # not take the pod out of rotation.
    return {"status": "ready"}


def handle_heartbeat(app: FastAPI, request: Request):
    """Serve ``GET /heartbeat`` after the scope check.

    Args:
        app: The FastAPI application.
        request: The inbound FastAPI request.

    Returns:
        object: The HTTP response.
    """
    context = route_scope(request, "GET", "/heartbeat")
    status, body = record_transport(
        transport_metrics_for(app),
        "heartbeat",
        "GET",
        lambda: op_heartbeat(
            app_context(app).node,
            cluster=app_context(app).cluster,
            headers=peer_headers(request),
            auth_context=context,
        ),
    )
    return respond(status, body)


def handle_metrics(app: FastAPI):
    """``GET /metrics`` — Prometheus text or legacy JSON.

    Args:
        app: The FastAPI application.

    Returns:
        object: Prometheus text, or the JSON snapshot.
    """
    refresh = app_context(app).refresh_metrics
    if refresh is not None:
        refresh()
    status, payload = op_metrics(app_context(app).node, app_context(app).metrics_registry)
    if status == 200 and isinstance(payload, tuple):
        text, headers = payload
        return PlainTextResponse(text, media_type=headers["media_type"])
    return respond(status, payload)


def metrics_json(app: FastAPI):
    """``GET /metrics.json`` — legacy JSON snapshot for the TUI.

    Args:
        app: The FastAPI application.

    Returns:
        object: The JSON metrics snapshot.
    """
    status, body = op_metrics(app_context(app).node)
    return respond(status, body)


__all__ = ["handle_heartbeat", "handle_metrics", "livez", "metrics_json", "readyz"]
