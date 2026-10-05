"""Expose the verified mTLS peer certificate CN to the ASGI app.

uvicorn terminates TLS but does not surface the client certificate
to the application. :class:`MTLSAuthenticator` authorizes callers by
their certificate CN, so without this module the only place it could
read the CN from is a request header, which any client can forge.

:class:`PeerCertH11Protocol` reads the CN from the handshake-verified
peer certificate when the connection is established and records it
in every request's ASGI scope under
``scope["extensions"]["tls"]["peer_cn"]``.
:func:`peer_headers_from_scope` builds the header map handed to the
authenticator: it drops any client-supplied ``x-ssl-client-cn`` and
substitutes the verified value.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from uvicorn.protocols.http.h11_impl import H11Protocol

PEER_CN_HEADER = "x-ssl-client-cn"


def peer_cn_from_transport(transport: asyncio.BaseTransport) -> str | None:
    """Return the CN of the verified peer certificate on ``transport``.

    Args:
        transport: The connection's asyncio transport.

    Returns:
        str | None: The certificate subject's commonName, or ``None``
        for plaintext connections and connections without a client
        certificate.
    """
    ssl_object = transport.get_extra_info("ssl_object")
    if ssl_object is None:
        return None
    # getpeercert() only returns a populated dict for a certificate
    # that passed verification against the configured CA bundle.
    cert = ssl_object.getpeercert()
    if not cert:
        return None
    for rdn in cert.get("subject", ()):
        for key, value in rdn:
            if key == "commonName" and value:
                return str(value)
    return None


class PeerCertH11Protocol(H11Protocol):
    """h11 protocol that records the verified peer CN in the ASGI scope."""

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        """Capture the peer CN once per connection and wrap the app."""
        super().connection_made(transport)
        peer_cn = peer_cn_from_transport(transport)
        inner_app = self.app

        async def app(scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") == "http":
                extensions = dict(scope.get("extensions") or {})
                tls = dict(extensions.get("tls") or {})
                tls["peer_cn"] = peer_cn
                extensions["tls"] = tls
                scope["extensions"] = extensions
            await inner_app(scope, receive, send)

        self.app = app


def peer_headers_from_scope(scope: Any, headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Return lower-cased request headers with a trusted peer CN.

    Args:
        scope: The ASGI scope of the request.
        headers: The raw request headers.

    Returns:
        dict[str, str]: Lower-cased headers. Any client-supplied
        ``x-ssl-client-cn`` is removed; the verified CN from the TLS
        handshake is inserted when one exists.
    """
    result = {k.lower(): v for k, v in headers if k.lower() != PEER_CN_HEADER}
    tls = ((scope or {}).get("extensions") or {}).get("tls") or {}
    peer_cn = tls.get("peer_cn")
    if peer_cn:
        result[PEER_CN_HEADER] = peer_cn
    return result


__all__ = [
    "PEER_CN_HEADER",
    "PeerCertH11Protocol",
    "peer_cn_from_transport",
    "peer_headers_from_scope",
]
