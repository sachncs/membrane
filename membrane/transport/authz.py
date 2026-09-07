"""Per-route scope wiring for the HTTP transport (Phase 3.1.1).

The v2.0 release defined :class:`membrane.auth.Authenticator` and
the scope hierarchy but left
:func:`membrane.auth.require_scope` unused outside ``/join``. The
v3.0.0 release wires the helper into every routed operation.

The :data:`ROUTE_SCOPES` table is the single source of truth: each
HTTP method+path pair maps to the minimum scope a caller must hold
to exercise the operation. ``enforce_route_scope`` is the helper
the FastAPI and stdlib route handlers call at the top of every
request: it authenticates the caller via the cluster's
:class:`~membrane.auth.Authenticator` and raises
:class:`~membrane.auth.AuthBackendError` (which the transport
translates to ``403``) if the caller is missing the required
scope.

Scope policy:

* ``/livez`` and ``/readyz`` are public (no scope) — they are
  process liveness probes that must succeed even before
  authentication is configured.
* Reads (``/retrieve``, ``/inventory``, ``/peers``,
  ``/heartbeat``) require ``read``.
* Writes (``/store``, ``/replicate``, ``/prefill``, ``/sync``,
  ``/gossip``, ``/join``, ``/leave``) require ``write``.
* Admin operations (``/delete``, ``/tombstone``, ``/purge``,
  ``/verify``) require ``admin``.

Routes not in :data:`ROUTE_SCOPES` default to ``read`` to fail
closed: an unlisted route is treated as a read-class operation,
and missing scope is rejected.
"""

from __future__ import annotations

from typing import Any

from membrane.auth import (
    AuthBackendError,
    AuthContext,
    AuthRequest,
    require_scope,
)

DEFAULT_SCOPE: str = "read"
"""Scope required for any route not listed in :data:`ROUTE_SCOPES`."""


ROUTE_SCOPES: dict[tuple[str, str], str] = {
    # Probes
    ("GET", "/livez"): "public",
    ("GET", "/readyz"): "public",
    # Observability
    ("GET", "/metrics"): "read",
    ("GET", "/metrics.json"): "read",
    ("GET", "/heartbeat"): "read",
    # Reads
    ("GET", "/retrieve"): "read",
    ("GET", "/inventory"): "read",
    ("GET", "/peers"): "read",
    # Writes
    ("POST", "/store"): "write",
    ("POST", "/replicate"): "write",
    ("POST", "/prefill"): "write",
    ("POST", "/sync"): "write",
    ("POST", "/gossip"): "write",
    ("POST", "/join"): "write",
    ("POST", "/leave"): "write",
    # Admin (Phase 3.2 wires the routes; the scope mapping is in
    # place so the first admin commit doesn't need a second pass).
    ("POST", "/delete"): "admin",
    ("POST", "/tombstone"): "admin",
    ("POST", "/purge"): "admin",
    ("POST", "/verify"): "admin",
}
"""(method, path) -> required scope. ``public`` means no auth check."""


def required_scope(method: str, path: str) -> str:
    """Return the scope required for ``method path``.

    Falls back to :data:`DEFAULT_SCOPE` for unlisted routes.
    Routes whose path starts with ``"/admin/"`` are
    unconditionally admin-scoped (the v3 admin surface
    always requires operator privilege).

    Args:
        method: HTTP method (uppercase).
        path: URL path, with or without a leading slash.

    Returns:
        str: The required scope name.
    """
    normalized = path if path.startswith("/") else f"/{path}"
    if normalized.startswith("/admin/"):
        return "admin"
    key = (method.upper(), normalized)
    return ROUTE_SCOPES.get(key, DEFAULT_SCOPE)


def _authenticate(
    authenticator: Any | None,
    method: str,
    path: str,
    headers: dict[str, str],
) -> AuthContext:
    """Authenticate ``request`` against ``authenticator``.

    Args:
        authenticator: The cluster's :class:`Authenticator` or
            ``None`` to bypass authentication (single-node
            deployments, tests).
        method: HTTP method.
        path: URL path.
        headers: Lowercased request headers.

    Returns:
        AuthContext: The caller's identity and scopes; an empty
        context (``scopes=frozenset()``) when no authenticator is
        configured.

    Raises:
        AuthBackendError: If the authenticator rejects the request.
    """
    if authenticator is None:
        return AuthContext(subject="", scopes=frozenset())
    request = AuthRequest(
        method=method.upper(),
        path=path,
        headers=dict(headers),
        client="",
    )
    return authenticator.authenticate(request)  # type: ignore[no-any-return]


def enforce_route_scope(
    authenticator: Any | None,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
) -> AuthContext:
    """Authenticate the caller and enforce the route's required scope.

    When ``authenticator`` is ``None`` the helper returns an empty
    context without performing any check. This preserves the
    single-node / test deployment path: a cluster that has not
    configured mTLS or API-key auth does not need scope checks,
    and the route surface remains open by configuration choice.
    When the cluster does configure an
    :class:`~membrane.auth.Authenticator`, the helper authenticates
    the caller and enforces the per-route scope from
    :data:`ROUTE_SCOPES`.

    Args:
        authenticator: The cluster's :class:`Authenticator` or
            ``None`` to bypass.
        method: HTTP method.
        path: URL path.
        headers: Lowercased request headers.

    Returns:
        AuthContext: The authenticated caller's context (or an
        empty context when no authenticator is configured).

    Raises:
        AuthBackendError: 401 if authentication fails; 403 if
            the caller is missing the required scope. Probes
            (``/livez`` and ``/readyz``) return an empty context
            without authenticating even when an authenticator is
            configured.
    """
    scope = required_scope(method, path)
    if scope == "public":
        return AuthContext(subject="", scopes=frozenset())
    if authenticator is None:
        return AuthContext(subject="", scopes=frozenset())
    headers = dict(headers or {})
    context = _authenticate(authenticator, method, path, headers)
    try:
        require_scope(context, scope)
    except AuthBackendError:
        raise
    return context


__all__ = [
    "DEFAULT_SCOPE",
    "ROUTE_SCOPES",
    "enforce_route_scope",
    "required_scope",
]
