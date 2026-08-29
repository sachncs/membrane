"""API-key-based :class:`Authenticator` implementation.

Keys are loaded from a ``keyfile``: a plain text file where each line is
``<api_key>:<subject>:<scope1>,<scope2>,...``. Lines starting with ``#``
are ignored.

Example keyfile::

    ak_live_admin:sre-bot:admin
    ak_live_writer:ingest-svc:write
    ak_live_reader:metrics-scraper:read
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from membrane.auth import AuthBackendError, AuthContext, Authenticator, AuthRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class APIKey:
    """A single API key entry.

    Attributes:
        key: The bearer token.
        subject: Stable identity for the caller.
        scopes: Set of scopes the key grants.
    """

    key: str
    subject: str
    scopes: frozenset[str]


def parse_keyfile(text: str) -> dict[str, APIKey]:
    """Parse a keyfile string into a ``key -> APIKey`` map.

    Args:
        text: Raw file contents.

    Returns:
        Dict mapping the bearer token to its :class:`APIKey` record.
    """
    result: dict[str, APIKey] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 3:
            logger.warning("ignoring malformed keyfile line: %r", raw_line)
            continue
        key = parts[0].strip()
        subject = parts[1].strip()
        scopes = frozenset(s.strip() for s in parts[2].split(",") if s.strip())
        result[key] = APIKey(key=key, subject=subject, scopes=scopes)
    return result


class APIKeyAuthenticator:
    """Authenticator that validates a bearer token against a keyfile.

    The Authorization header is expected in the form ``Bearer <key>``. Any
    other shape (no header, wrong scheme, unknown key) is rejected with
    :class:`AuthBackendError`.
    """

    def __init__(self, keyfile_text: str) -> None:
        """Initialize with the raw keyfile text.

        Args:
            keyfile_text: Contents of the keyfile; see module docstring
                for the expected format.
        """
        self.keys = parse_keyfile(keyfile_text)

    def authenticate(self, request: AuthRequest) -> AuthContext:
        """Authenticate a request via its Authorization header.

        Args:
            request: The transport-agnostic request.

        Returns:
            AuthContext: The caller's identity and scopes.

        Raises:
            AuthBackendError: If the header is missing, malformed, or the key
                is unknown.
        """
        header = request.headers.get("authorization", "")
        if not header:
            raise AuthBackendError("unauthorized")
        parts = header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthBackendError("unauthorized")
        key = parts[1].strip()
        record = self.keys.get(key)
        if record is None:
            raise AuthBackendError("unauthorized")
        return AuthContext(subject=record.subject, scopes=record.scopes)


class NoopAuthenticator:
    """Authenticator that accepts every request with no scopes.

    Used by tests and by transports that bypass auth (e.g., ``/livez``).
    """

    def authenticate(self, request: AuthRequest) -> AuthContext:
        """Return an empty context for any request."""
        return AuthContext(subject="", scopes=frozenset())


__all__ = ["APIKey", "APIKeyAuthenticator", "NoopAuthenticator"]


def ensure_runtime_checkable() -> None:
    """Sanity-check that both implementations satisfy the :class:`Authenticator` protocol."""
    # These are no-ops at runtime; they exist purely to give a clear error
    # at import time if a bug regresses the protocol implementation.
    assert isinstance(APIKeyAuthenticator(""), Authenticator)
    assert isinstance(NoopAuthenticator(), Authenticator)


ensure_runtime_checkable()


def ignore_unused(_: Any) -> None:
    """Suppress unused-import warnings for type-checker-only imports."""
    return None
