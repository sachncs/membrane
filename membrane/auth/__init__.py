"""Authenticator protocol for inbound request authentication.

The :class:`Authenticator` protocol is the single point where any transport
decides whether a request is allowed. Concrete implementations plug in
API-key or mTLS-based authentication; tests use :class:`NoopAuthenticator`
to skip auth entirely.

Each transport (HTTP, FastAPI, gRPC) wraps its request, calls
:meth:`Authenticator.authenticate`, and either accepts the call or rejects
it with the appropriate status code. The result is an :class:`AuthContext`
that carries the caller's identity and the granted scopes, which downstream
handlers use for scope checks via :func:`require_scope`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AuthContext:
    """The authenticated caller's identity and granted scopes.

    Attributes:
        subject: A stable identifier for the caller (e.g., API key ID,
            client certificate CN). Empty string when authentication
            is bypassed (e.g., for ``/livez``).
        scopes: Immutable set of scope strings the caller is allowed
            to exercise. See :data:`SCOPES` for the canonical list.
    """

    subject: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class AuthRequest:
    """Transport-agnostic description of an inbound request.

    Carries just enough information for any :class:`Authenticator`
    implementation to make a decision. Transports adapt their native
    request objects into this shape.

    Attributes:
        method: HTTP method or RPC name (uppercase by convention).
        path: URL path or RPC service identifier.
        headers: Headers / metadata as ``dict[str, str]``.
        client: Optional client identifier (e.g., peer IP) for rate
            limiting or audit logs.
    """

    method: str
    path: str
    headers: dict[str, str]
    client: str = ""


class AuthBackendError(Exception):
    """Raised by authenticators when the request is rejected."""


@runtime_checkable
class Authenticator(Protocol):
    """Protocol for inbound authentication.

    Implementations must:
        * Return an :class:`AuthContext` (with ``subject`` and ``scopes``)
          on success.
        * Raise :class:`AuthBackendError` with a generic message on failure.
          The transport translates this to the appropriate status code.
    """

    def authenticate(self, request: AuthRequest) -> AuthContext:
        """Authenticate ``request``.

        Args:
            request: The transport-agnostic request description.

        Returns:
            AuthContext: The caller's identity and granted scopes.

        Raises:
            AuthBackendError: If authentication fails.
        """
        ...


SCOPES: dict[str, frozenset[str]] = {
    "read": frozenset(),
    "write": frozenset({"read"}),
    "admin": frozenset({"read", "write"}),
}
"""Canonical scope hierarchy. ``admin`` implies ``write`` and ``read``."""


def require_scope(context: AuthContext, scope: str) -> None:
    """Raise :class:`AuthBackendError` if ``context`` does not have ``scope``.

    Checks the ``SCOPES`` hierarchy so that holding ``admin`` satisfies a
    ``read`` check, and so on.

    Args:
        context: The authenticated caller's context.
        scope: The scope the caller must hold.

    Raises:
        AuthBackendError: If the caller does not hold ``scope``.
    """
    if scope in context.scopes:
        return
    # Hierarchical expansion: admin implies write implies read.
    for granted in context.scopes:
        if scope in SCOPES.get(granted, frozenset()):
            return
    raise AuthBackendError(f"missing required scope: {scope}")


__all__ = [
    "AuthContext",
    "AuthBackendError",
    "AuthRequest",
    "Authenticator",
    "SCOPES",
    "require_scope",
]
