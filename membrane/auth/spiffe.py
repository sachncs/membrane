"""Authorize callers by SPIFFE ID (the ``spiffe://`` URI in their SVID).

With ``--tls-spiffe-socket``, peers and clients present X.509 SVIDs. The
listener verifies them against the trust bundle, and the verified URI
is the caller's identity (see
:func:`membrane.transport.tls_protocol.peer_cn_from_transport`).
:class:`SPIFFEAuthenticator` grants each allowed ID its configured
scopes; any other ID is rejected.
"""

import logging

from membrane.auth import AuthBackendError, AuthContext, AuthRequest
from membrane.transport.tls import parse_peer_cn_header

logger = logging.getLogger(__name__)


def parse_id_scopes(entries: tuple[str, ...] | list[str]) -> dict[str, frozenset[str]]:
    """Parse ``spiffe://td/path=scope,scope`` entries.

    Args:
        entries: Allow-list entries.

    Returns:
        dict[str, frozenset[str]]: SPIFFE ID to granted scopes.

    Raises:
        ValueError: On an entry that is not ``spiffe://...=scopes``.
    """
    allowed: dict[str, frozenset[str]] = {}
    for entry in entries:
        spiffe_id, sep, scopes = entry.partition("=")
        if not sep or not spiffe_id.startswith("spiffe://") or not scopes:
            raise ValueError(f"expected spiffe://trust-domain/path=scope[,scope], got {entry!r}")
        allowed[spiffe_id] = frozenset(s.strip() for s in scopes.split(",") if s.strip())
    return allowed


class SPIFFEAuthenticator:
    """Authenticate by verified SPIFFE ID against an allow-list.

    Attributes:
        allowed: SPIFFE ID to granted scopes.
    """

    def __init__(self, allowed: dict[str, frozenset[str]]) -> None:
        """Create the authenticator.

        Args:
            allowed: SPIFFE ID to granted scopes.
        """
        self.allowed = allowed

    def authenticate(self, request: AuthRequest) -> AuthContext:
        """Admit a caller whose verified SPIFFE ID is allowed.

        Args:
            request: The transport-agnostic request.

        Returns:
            AuthContext: ``subject`` is the SPIFFE ID; scopes come from the
            allow-list.

        Raises:
            AuthBackendError: When the caller presented no SVID or an ID
                that is not allowed.
        """
        identity = parse_peer_cn_header(request.headers)
        if not identity:
            raise AuthBackendError("SPIFFE SVID required")
        scopes = self.allowed.get(identity)
        if not scopes:
            logger.warning("Rejecting %s %s from %s: SPIFFE ID not allowed", request.method, request.path, identity)
            raise AuthBackendError("SPIFFE ID not allowed")
        return AuthContext(subject=identity, scopes=scopes)


__all__ = ["SPIFFEAuthenticator", "parse_id_scopes"]
