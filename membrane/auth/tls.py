"""TLS configuration dataclass for transport encryption.

Transport modules (``membrane.transport.http``, ``membrane.transport.fastapi``,
``membrane.transport.grpc``) all read a :class:`TLSConfig` from the
:class:`Server` constructor and apply it polymorphically via the
``ssl_context`` property.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TLSConfig:
    """TLS configuration for inbound transport encryption.

    Attributes:
        keyfile: Path to the PEM-encoded private key.
        certfile: Path to the PEM-encoded certificate (or chain).
        cafile: Optional path to a CA bundle for verifying client
            certificates (mTLS). When set, ``require_client_cert``
            should also be ``True``.
        require_client_cert: If ``True``, mTLS is enforced (server
            verifies the client certificate). If ``False``, only
            server-side TLS is applied and client certs are optional.
    """

    keyfile: str
    certfile: str
    cafile: str | None = None
    require_client_cert: bool = False

    def __post_init__(self) -> None:
        for attr in ("keyfile", "certfile"):
            path = getattr(self, attr)
            if not Path(path).exists():
                raise FileNotFoundError(f"TLS {attr} not found: {path}")
        if self.cafile is not None and not Path(self.cafile).exists():
            raise FileNotFoundError(f"TLS cafile not found: {self.cafile}")

    def ssl_context(self) -> ssl.SSLContext:
        """Build an :class:`ssl.SSLContext` configured from this TLSConfig.

        Returns:
            ssl.SSLContext: Configured for server-side TLS, optionally
            with mutual TLS verification.
        """
        purpose = ssl.Purpose.CLIENT_AUTH if self.require_client_cert else ssl.Purpose.SERVER_AUTH
        ctx = ssl.create_default_context(purpose)
        ctx.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
        if self.cafile is not None:
            ctx.load_verify_locations(cafile=self.cafile)
        if self.require_client_cert:
            ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx


__all__ = ["TLSConfig"]
