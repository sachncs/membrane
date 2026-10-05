"""Shared helpers for the HTTP routes: authentication, metrics, and responses."""

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from membrane.auth import AuthContext, AuthForbiddenError
from membrane.metrics import TransportMetrics
from membrane.serialization import JsonDict
from membrane.transport.authz import enforce_route_scope
from membrane.transport.context import app_context
from membrane.transport.tls_protocol import peer_headers_from_scope


def authenticator_for(app: FastAPI) -> object | None:
    """Return the cluster's authenticator from app.state, if any.

    When ``app_context(app).authenticator`` was populated by the Server at
    startup with an
    :class:`~membrane.auth.mtls.MTLSAuthenticator`, the route
    handler calls :func:`membrane.transport.authz.enforce_route_scope`
    against it. Absent the cluster is treated as non-mTLS
    (single-node deployments).

    Args:
        app: The FastAPI application.

    Returns:
        object | None: The cluster's authenticator from app.state, if any.
    """
    return app_context(app).authenticator


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

    Args:
        request: The inbound FastAPI request.

    Returns:
        dict[str, str]: Lower-cased headers with a trusted peer CN.
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


def auth_error_response(_request: Request, exc: Exception) -> Response:
    """Translate authentication / authorization failures to 401 / 403.

    The body is deliberately generic so a probe cannot distinguish
    an unknown key from a known key with the wrong scope beyond the
    status code itself.

    Args:
        exc: The authentication or authorization error.

    Returns:
        Response: A 401 or 403 JSON response.
    """
    if isinstance(exc, AuthForbiddenError):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return JSONResponse(
        {"error": "unauthorized"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def transport_metrics_for(app: FastAPI) -> TransportMetrics | None:
    """Return the cluster's TransportMetrics from app.state, if any.

    The :class:`Server` populates ``app_context(app).transport_metrics``
    at construction; transport handlers use the helper to
    record every op invocation.

    Args:
        app: The FastAPI application.

    Returns:
        TransportMetrics | None: The cluster's TransportMetrics from
        app.state, if any.
    """
    return app_context(app).transport_metrics


def respond(status: int, body: JsonDict | tuple[str, dict[str, str]] | object) -> Response:
    """Translate an operation's ``(status, body)`` to a FastAPI response.

    Args:
        status: HTTP status code.
        body: Response body (JSON, or Prometheus text).

    Returns:
        Response: The JSON (or text) response.
    """
    if status == 200:
        return JSONResponse(body)
    return JSONResponse(body, status_code=status)


def respond_with_retry_after(status: int, body: JsonDict, retry_after: int) -> Response:
    """Build a JSON response carrying a ``Retry-After`` header.

    Args:
        status: HTTP status code.
        body: Response body.
        retry_after: Seconds the client should wait before retrying.

    Returns:
        Response: A JSON response carrying a ``Retry-After`` header.
    """
    return JSONResponse(
        body,
        status_code=status,
        headers={"Retry-After": str(retry_after)},
    )


async def read_limited_body(request: Request, limit: int) -> bytes | None:
    """Read the request body, giving up once it exceeds ``limit`` bytes.

    Args:
        request: The inbound request.
        limit: Maximum body size in bytes.

    Returns:
        bytes | None: The body, or ``None`` when it is too large.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        return None
    parts: list[bytes] = []
    total = 0
    async for part in request.stream():
        total += len(part)
        if total > limit:
            return None
        parts.append(part)
    return b"".join(parts)


__all__ = [
    "auth_error_response",
    "authenticator_for",
    "peer_headers",
    "read_limited_body",
    "respond",
    "respond_with_retry_after",
    "route_scope",
    "transport_metrics_for",
]
