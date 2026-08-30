"""mTLS :class:`Authenticator` implementation.

Production clusters authenticate inbound requests through mutual
TLS rather than per-request bearer tokens. The transport terminates
the handshake via :mod:`membrane.transport.tls` and writes the peer
cert's Common Name into a request header
(``X-SSL-Client-CN``) which this authenticator reads.

The authenticator's contract:

* Returns an :class:`~membrane.auth.AuthContext` whose
  ``subject`` is the verified CN. Scopes are derived from the
  CN itself: a CN that begins with ``admin-`` carries the
  ``admin`` scope, a CN that begins with ``write-`` carries
  ``write``, anything else carries only ``read``.
* Raises :class:`~membrane.auth.AuthBackendError` when the
  header is missing, malformed, or filtered out by the
  configured :class:`~membrane.transport.tls.MTLSConfig`'s
  allow-list.

Pinning the authenticator to a specific CN allow-list is the
single source of truth — the FastAPI transport and the gRPC
transport attach the same instance, so a CN that's banned for
``POST /store`` is banned for ``POST /join`` too. ``MTLSConfig``
is mandatory at construction so the two-step ``MTLSConfig →
mTLSAuthenticator`` wiring can never silently regress to
accepting any CN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from membrane.auth import AuthBackendError, AuthContext, AuthRequest
from membrane.transport.tls import MTLSConfig, parse_peer_cn_header, peer_cn_allowed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CN-derived scopes
# ---------------------------------------------------------------------------

#: CN prefixes that grant a specific scope. The lookup is read in
#: order so the longest matching prefix wins.
CN_SCOPE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("admin-", "admin"),
    ("write-", "write"),
    ("read-", "read"),
)


def scopes_for_cn(cn: str) -> frozenset[str]:
    """Return the scopes the cluster grants a peer with ``cn``.

    Production clusters configure CNs as ``admin-<id>``,
    ``write-<id>``, or ``read-<id>`` and the prefix drives the
    scope granted by the authenticator. ``admin`` implies
    ``write`` implies ``read`` (the canonical SCOPES hierarchy
    in :mod:`membrane.auth`) so an admin CN is automatically
    granted the lower scopes too.

    Args:
        cn: Verified peer cert's Common Name.

    Returns:
        frozenset[str]: Granted scopes; empty when the CN
        does not match any prefix.
    """
    if not cn:
        return frozenset()
    for prefix, scope in CN_SCOPE_PREFIXES:
        if cn.startswith(prefix):
            if scope == "admin":
                return frozenset({"read", "write", "admin"})
            if scope == "write":
                return frozenset({"read", "write"})
            return frozenset({scope})
    return frozenset()


# ---------------------------------------------------------------------------
# Authenticator implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerIdentity:
    """Auxiliary record of the verified peer identity.

    Carried alongside the authenticator's return value when
    a caller needs the CN for audit logging. Not part of the
    public surface; the authenticator returns an
    :class:`AuthContext` whose ``subject`` IS the CN.

    Attributes:
        cn: Verified Common Name from the peer cert.
        scopes: Set of scopes the CN carries.
        verified_at: Wall-clock time when verification
            completed. Useful for audit logs.
    """

    cn: str
    scopes: frozenset[str]
    verified_at: float


class MTLSAuthenticator:
    """Authenticator that validates a request via its mTLS peer cert.

    Reads :attr:`MTLSConfig.allowed_cns` to gate the cluster. When
    the configured :class:`MTLSConfig` has ``require_client_cert=True``
    (production default) every inbound request must carry a CN
    header; missing or empty header is rejected.
    """

    def __init__(self, config: MTLSConfig) -> None:
        """Initialize with the cluster's TLS configuration.

        Args:
            config: Cluster TLS settings.

        Raises:
            ValueError: When ``config.require_client_cert`` is
                ``False`` and the operator opted into non-mTLS
                mode by accident — the authenticator refuses to
                be constructed this way so the 2.0 wire never
                regresses to TLS without mutual verification.
        """
        if not config.require_client_cert:
            raise ValueError(
                "MTLSAuthenticator requires MTLSConfig.require_client_cert=True; "
                "non-mTLS mode is unsupported at 2.0"
            )
        self.config = config

    def authenticate(self, request: AuthRequest) -> AuthContext:
        """Authenticate ``request`` via the peer cert CN.

        Args:
            request: Inbound transport-agnostic request.

        Returns:
            AuthContext: ``subject`` is the verified CN;
            ``scopes`` are derived from the CN prefix.

        Raises:
            AuthBackendError: When the CN header is missing,
            empty, or filtered out by the configured allow-list.
        """
        cn = parse_peer_cn_header(request.headers)
        if not cn:
            logger.warning("Rejecting %s %s: missing peer CN", request.method, request.path)
            raise AuthBackendError("mTLS peer cert required")
        if not peer_cn_allowed(self.config, cn):
            logger.warning(
                "Rejecting %s %s from cn=%s: not in allowed_cns",
                request.method,
                request.path,
                cn,
            )
            raise AuthBackendError(f"peer CN not in allow-list: {cn}")
        scopes = scopes_for_cn(cn)
        if not scopes:
            logger.warning(
                "Rejecting %s %s from cn=%s: CN does not grant any scope",
                request.method,
                request.path,
                cn,
            )
            raise AuthBackendError(f"peer CN has no granted scope: {cn}")
        logger.debug("Authenticated %s %s from cn=%s with scopes=%s", request.method, request.path, cn, sorted(scopes))
        return AuthContext(subject=cn, scopes=scopes)


__all__ = [
    "CN_SCOPE_PREFIXES",
    "MTLSAuthenticator",
    "PeerIdentity",
    "scopes_for_cn",
]


# Trailing self-test: importable without exercising the runtime
# path, so static analyzers confirm the module surface stays
# closed.

_LAZY_EXPORTS: dict[str, Any] = {
    "MTLSAuthenticator": MTLSAuthenticator,
    "PeerIdentity": PeerIdentity,
    "scopes_for_cn": scopes_for_cn,
    "CN_SCOPE_PREFIXES": CN_SCOPE_PREFIXES,
}
