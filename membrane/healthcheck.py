"""Container health check: ``python -m membrane.healthcheck``.

Exits 0 when the local server answers ``/livez``, 1 otherwise. It reads
the same ``MEMBRANE_*`` variables as ``membrane serve``: when the
listener uses mTLS, it connects over HTTPS and presents the node's own
certificate. The image needs no curl.
"""

import os
import ssl
import sys
import urllib.request

TIMEOUT_SEC = 4.0


def livez_url() -> tuple[str, ssl.SSLContext | None]:
    """Return the local ``/livez`` URL and the TLS context to reach it.

    Returns:
        tuple[str, ssl.SSLContext | None]: The URL, and a client context
        when ``MEMBRANE_TLS_CERT_FILE`` is set (``None`` for plain HTTP).
    """
    port = os.environ.get("MEMBRANE_PORT", "8080")
    cert = os.environ.get("MEMBRANE_TLS_CERT_FILE", "")
    if not cert:
        return f"http://127.0.0.1:{port}/livez", None
    context = ssl.create_default_context(cafile=os.environ.get("MEMBRANE_TLS_CA_FILE") or None)
    # The certificate names the node's advertised host, not 127.0.0.1.
    context.check_hostname = False
    context.load_cert_chain(cert, os.environ.get("MEMBRANE_TLS_KEY_FILE") or None)
    return f"https://127.0.0.1:{port}/livez", context


def main() -> int:
    """Probe ``/livez`` once.

    Returns:
        int: ``0`` when healthy, ``1`` otherwise.
    """
    url, context = livez_url()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SEC, context=context) as response:
            return 0 if response.status == 200 else 1
    except OSError, ValueError:
        return 1


__all__ = ["TIMEOUT_SEC", "livez_url", "main"]

if __name__ == "__main__":
    sys.exit(main())
