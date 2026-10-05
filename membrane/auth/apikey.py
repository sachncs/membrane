"""API-key-based :class:`Authenticator` implementation.

Keys are loaded from a ``keyfile`` with one key per line, in either form::

    sha256:<64 hex digest of the key>:<subject>:<scope1>,<scope2>,...
    <api_key>:<subject>:<scope1>,<scope2>,...

The hashed form is recommended: a leaked keyfile then reveals no usable
credentials. ``membrane keys generate`` prints a fresh key together with
its hashed line. Plaintext lines still work, but a warning is logged.
Lines starting with ``#`` are ignored.

Example keyfile::

    sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08:sre-bot:admin
    ak_live_writer:ingest-svc:write

Only SHA-256 digests are held in memory. A presented token is hashed and
compared against them in constant time.
"""

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass

from membrane.auth import AuthBackendError, AuthContext, AuthRequest

logger = logging.getLogger(__name__)

HASH_PREFIX = "sha256"
KEY_PREFIX = "mbr_"


def hash_key(key: str) -> str:
    """Return the hex SHA-256 digest a keyfile stores for ``key``.

    API keys are long random tokens, so a fast unsalted hash is enough:
    there is no low-entropy secret to brute-force.

    Args:
        key: The bearer token.

    Returns:
        str: 64 lowercase hex characters.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key(subject: str, scopes: list[str]) -> tuple[str, str]:
    """Create a random API key and its hashed keyfile line.

    Args:
        subject: Stable identity of the caller.
        scopes: Scopes the key grants (e.g. ``["read", "write"]``).

    Returns:
        tuple[str, str]: ``(key, keyfile_line)``. Hand the key to the
        client; put only the line in the keyfile.
    """
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, f"{HASH_PREFIX}:{hash_key(key)}:{subject}:{','.join(scopes)}"


@dataclass(frozen=True, slots=True)
class APIKey:
    """A single API key entry.

    Attributes:
        digest: Hex SHA-256 digest of the bearer token.
        subject: Stable identity for the caller.
        scopes: Set of scopes the key grants.
    """

    digest: str
    subject: str
    scopes: frozenset[str]


def parse_line(raw_line: str) -> tuple[APIKey, bool] | None:
    """Parse one keyfile line.

    Args:
        raw_line: The line, without its newline.

    Returns:
        tuple[APIKey, bool] | None: The record and whether the line held a
        plaintext key; ``None`` for blank, comment, or malformed lines.
    """
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [part.strip() for part in line.split(":")]
    match parts:
        case [prefix, digest, subject, scopes] if prefix == HASH_PREFIX:
            plaintext = False
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                logger.warning("ignoring keyfile line with a malformed sha256 digest")
                return None
            digest = digest.lower()
        case [prefix, *_] if prefix == HASH_PREFIX:
            logger.warning("ignoring malformed sha256 keyfile line")
            return None
        case [key, subject, scopes, *_] if key:
            plaintext = True
            digest = hash_key(key)
        case _:
            logger.warning("ignoring malformed keyfile line")
            return None
    # An empty subject would read as "unauthenticated" downstream and
    # bypass the per-tenant read check, so such lines are rejected.
    if not subject:
        logger.warning("ignoring keyfile line with an empty subject")
        return None
    scope_set = frozenset(s.strip() for s in scopes.split(",") if s.strip())
    return APIKey(digest=digest, subject=subject, scopes=scope_set), plaintext


def parse_keyfile(text: str) -> dict[str, APIKey]:
    """Parse a keyfile into a ``digest -> APIKey`` map.

    Args:
        text: Raw file contents.

    Returns:
        dict[str, APIKey]: Records keyed by the SHA-256 digest of their key.
    """
    result: dict[str, APIKey] = {}
    plaintext_lines = 0
    for raw_line in text.splitlines():
        parsed = parse_line(raw_line)
        if parsed is None:
            continue
        record, plaintext = parsed
        plaintext_lines += plaintext
        result[record.digest] = record
    if plaintext_lines:
        logger.warning(
            "keyfile holds %s plaintext key(s); store 'sha256:<digest>:...' lines instead (membrane keys generate)",
            plaintext_lines,
        )
    return result


class APIKeyAuthenticator:
    """Authenticator that validates a bearer token against a keyfile.

    The Authorization header is expected in the form ``Bearer <key>``. Any
    other shape (no header, wrong scheme, unknown key) is rejected with
    :class:`AuthBackendError`.

    Attributes:
        keys: Records keyed by the SHA-256 digest of their key.
    """

    def __init__(self, keyfile_text: str) -> None:
        """Initialize with the raw keyfile text.

        Args:
            keyfile_text: Contents of the keyfile; see module docstring
                for the expected format.
        """
        self.keys = parse_keyfile(keyfile_text)

    def lookup(self, key: str) -> APIKey | None:
        """Return the record for a bearer token, comparing digests in constant time.

        Args:
            key: The presented bearer token.

        Returns:
            APIKey | None: The matching record, or ``None``.
        """
        digest = hash_key(key)
        found: APIKey | None = None
        for stored, record in self.keys.items():
            if hmac.compare_digest(stored, digest):
                found = record
        return found

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
        record = self.lookup(parts[1].strip())
        if record is None:
            raise AuthBackendError("unauthorized")
        return AuthContext(subject=record.subject, scopes=record.scopes)


class NoopAuthenticator:
    """Authenticator that accepts every request with no scopes.

    Used by tests and by transports that bypass auth (e.g., ``/livez``).
    """

    def authenticate(self, request: AuthRequest) -> AuthContext:
        """Return an empty context for any request.

        Args:
            request: The transport-agnostic request.

        Returns:
            AuthContext: An empty context for any request.
        """
        return AuthContext(subject="", scopes=frozenset())


__all__ = [
    "HASH_PREFIX",
    "KEY_PREFIX",
    "APIKey",
    "APIKeyAuthenticator",
    "NoopAuthenticator",
    "generate_key",
    "hash_key",
    "parse_keyfile",
    "parse_line",
]
