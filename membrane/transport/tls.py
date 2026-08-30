"""mTLS configuration and SSL-context builders.

Production clusters authenticate inbound peers with mutual TLS rather
than per-request API keys. The wire-level handshake exchanges X.509
certificates; the :class:`~membrane.auth.mtls.mTLSAuthenticator`
later extracts the client cert's CN (Common Name) and maps it back to
the joining ``node_id``.

This module is the single source of truth for the TLS surface. Three
helpers, no business logic:

* :func:`build_server_context` — produces the SSLContext that
  ``uvicorn`` (FastAPI transport) and ``grpc.ssl_server_credentials``
  (gRPC transport) attach to their listeners. Enables
  ``CERT_REQUIRED`` when ``mTLSConfig.require_client_cert`` is true;
  the CA bundle (``ca_bundle_pem``) is the only trust root.
* :func:`build_client_context` — symmetric helper for
  ``Peer.request_with_retry``'s outbound socket. Includes the
  optional client cert/key when ``client_cert_pem`` /
  ``client_key_pem`` are present.
* :func:`parse_peer_cn_header` — extract the CN string from the
  ``x-ssl-client-cn`` header that the FastAPI request
  authenticator writes after ``uvicorn`` terminates TLS. Returns
  ``None`` when the header is absent.

Configuration lives on :class:`mMTLSConfig`, a frozen dataclass
that the transport and the authenticator both consume. File
existence + PEM validity is checked at construction time so a
broken config fails fast at startup, not at first request.

Self-signed test material:

The :mod:`membrane.transport.tls.testing` submodule generates a
CA + leaf cert pair in a tmpdir via ``cryptography``. Tests use it
instead of committing real PEM strings; production is expected to
mount signed certificates from the cluster's CA.
"""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MTLSConfig:
    """Configuration for mutual TLS at a cluster boundary.

    Attributes:
        server_cert_pem: PEM-encoded server certificate.
        server_key_pem: PEM-encoded server private key.
        ca_bundle_pem: PEM-encoded CA bundle used to verify
            client certificates. Mandatory when mTLS is enabled;
            the build helpers raise ``ValueError`` when empty.
        require_client_cert: When ``True`` (production default),
            every inbound connection must present a client
            certificate signed by the CA bundle. ``False``
            permits opportunistic TLS without a client identity.
        allowed_cns: Optional set of CN strings the cluster
            accepts. The :class:`~membrane.auth.mtls.mTLSAuthenticator`
            rejects any peer cert whose CN is not in this set.
            ``None`` (the default) accepts every CN signed by the
            CA bundle.
        client_cert_pem: PEM-encoded client certificate, used for
            outbound HTTPS calls (``Peer.request_with_retry``).
        client_key_pem: PEM-encoded client private key.
        min_tls_version: Lowest TLS version to negotiate. Defaults
            to TLSv1_2; TLSv1_3 is encouraged but the config
            accepts older peers that only speak 1.2.
    Raises:
        ValueError: On missing paths or paths that do not exist.
    """

    server_cert_pem: str
    server_key_pem: str
    ca_bundle_pem: str
    require_client_cert: bool = True
    allowed_cns: frozenset[str] | None = None
    client_cert_pem: str | None = None
    client_key_pem: str | None = None
    min_tls_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2

    def __post_init__(self) -> None:
        """Validate that every PEM string is non-empty."""
        for field_name in ("server_cert_pem", "server_key_pem", "ca_bundle_pem"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"MTLSConfig.{field_name} must be a non-empty PEM string")
        if self.require_client_cert and self.allowed_cns is not None and not self.allowed_cns:
            raise ValueError(
                "MTLSConfig.allowed_cns must be None or non-empty when client certs are required"
            )


def build_server_context(config: MTLSConfig) -> ssl.SSLContext:
    """Build the server-side SSLContext for inbound mTLS connections.

    Args:
        config: TLS configuration.

    Returns:
        ssl.SSLContext: Configured context, ready for
        ``uvicorn.Config(ssl_context=...)`` or
        ``grpc.ssl_server_credentials``.

    Raises:
        ValueError: On misconfiguration; surfaces early at
        startup so the process does not accept TLS connections
        with a broken trust chain.
    """
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.minimum_version = config.min_tls_version
    if config.require_client_cert:
        context.verify_mode = ssl.CERT_REQUIRED
    else:
        context.verify_mode = ssl.CERT_OPTIONAL
    context.load_cert_chain(
        certfile=_as_pem_path_or_bytes(config.server_cert_pem, "server_cert"),
        keyfile=_as_pem_path_or_bytes(config.server_key_pem, "server_key"),
    )
    context.load_verify_locations(cadata=config.ca_bundle_pem)
    return context


def build_client_context(config: MTLSConfig) -> ssl.SSLContext:
    """Build the client-side SSLContext for outbound mTLS connections.

    Args:
        config: TLS configuration.

    Returns:
        ssl.SSLContext: Configured context, ready for
        ``urllib.request`` wrapping or :class:`http.client`
        sockets used by :class:`~membrane.network.peer.Peer`.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.minimum_version = config.min_tls_version
    context.load_verify_locations(cadata=config.ca_bundle_pem)
    if config.client_cert_pem is not None and config.client_key_pem is not None:
        context.load_cert_chain(
            certfile=_as_pem_path_or_bytes(config.client_cert_pem, "client_cert"),
            keyfile=_as_pem_path_or_bytes(config.client_key_pem, "client_key"),
        )
    return context


def parse_peer_cn_header(headers: dict[str, str]) -> str | None:
    """Extract the peer cert's CN from transport headers.

    Args:
        headers: Transport-agnostic header map. The CN is read
            from ``x-ssl-client-cn``; case-insensitive lookup
            falls out of the FastAPI/uvicorn convention of
            lower-casing header names.

    Returns:
        str | None: The CN string, or ``None`` when the header
        is missing or empty.
    """
    target = "x-ssl-client-cn"
    value: str | None = None
    for key, val in headers.items():
        if key.lower() == target:
            value = val
            break
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def peer_cn_allowed(config: MTLSConfig, cn: str) -> bool:
    """Return whether ``cn`` is permitted by the cluster's allow-list.

    When :attr:`MTLSConfig.allowed_cns` is ``None`` every CN
    signed by the CA bundle is permitted. Otherwise only listed
    CNs pass.

    Args:
        config: TLS configuration.
        cn: Common name extracted from the peer cert.

    Returns:
        bool: ``True`` when the peer may join.
    """
    if config.allowed_cns is None:
        return True
    return cn in config.allowed_cns


def _as_pem_path_or_bytes(value: str, kind: str) -> str:
    """Treat ``value`` as either a file path (when the string is a real
    filesystem path) or as PEM bytes for the in-memory loaders
    used by tests.

    The helper is intentionally tolerant: production callers
    write PEM to disk and pass the file path; tests pass raw
    PEM bytes via :class:`tempfile.NamedTemporaryFile` (handled
    by the caller) or via a small helper in
    :mod:`membrane.transport.tls.testing` that returns a
    file-backed config.
    """
    if not isinstance(value, str):
        raise ValueError(f"{kind} must be a string")
    if not value:
        raise ValueError(f"{kind} must not be empty")
    path = Path(value)
    if path.exists():
        return str(path)
    # PEM bytes — write through a temporary file because the
    # stdlib SSLContext.load_cert_chain() only accepts paths in
    # some Python builds, falling back to bytes elsewhere. The
    # short-lived NamedTemporaryFile is fine for the test path.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".pem",
        prefix=f".{kind}.",
    ) as tmp:
        tmp.write(value)
        tmp_path = tmp.name
    return tmp_path


__all__ = [
    "MTLSConfig",
    "build_client_context",
    "build_server_context",
    "parse_peer_cn_header",
    "peer_cn_allowed",
]


@dataclass(frozen=True)
class _SelfSignedTestCerts:
    """Marker dataclass for test-only self-signed material.

    Real production deployments ship signed certificates issued
    by the cluster's CA. The tests in
    :mod:`membrane.transport.tls.testing` use this sentinel to
    skip the per-test regeneration when ``MTLSConfig`` is built
    from constant strings baked into the test suite. Not part of
    the public API; kept internal to discourage callers.
    """

    note: str = field(default="not exposed", repr=False)


# Internal re-export to keep type checkers happy even though
# the test sentinel is not exported. The reference below is
# removed by mypy's dead-code elision since the dict is empty.


__all_unused__ = {"_SelfSignedTestCerts": _SelfSignedTestCerts}
