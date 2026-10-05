"""SPIFFE workload identity: TLS material from the SPIFFE Workload API.

With ``--tls-spiffe-socket``, the node's certificate, key, and trust
bundle come from the Workload API (SPIRE agent), not from files. SVIDs
are short-lived, so the server re-fetches them every
:data:`REFRESH_INTERVAL_SEC` and serves a renewed SVID without a
restart. Requires ``pip install membrane[tls-spiffe]``.
"""

import logging
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import serialization

from membrane.transport.tls import MTLSConfig

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SEC = 300.0


@dataclass
class SPIFFEConfig:
    """SPIFFE workload identity configuration.

    Attributes:
        socket_path: Workload API socket (``unix:///run/spire/agent.sock``
            style or a filesystem path).
        allowed_ids: SPIFFE IDs accepted as peers or clients.
    """

    socket_path: str = "/run/spiffe/workload-api.sock"
    allowed_ids: frozenset[str] = frozenset()


def to_pem(cert: Any) -> str:
    """Encode a :mod:`cryptography` certificate as PEM.

    Args:
        cert: The certificate.

    Returns:
        str: PEM text.
    """
    return cert.public_bytes(serialization.Encoding.PEM).decode()


class SPIFFEClient:
    """Fetches the node's X.509 SVID and trust bundle from the Workload API."""

    def __init__(self, config: SPIFFEConfig | None = None) -> None:
        """Create a client.

        Args:
            config: Socket and allowed IDs.
        """
        self.config = config or SPIFFEConfig()

    def fetch_mtls_config(self) -> MTLSConfig:
        """Fetch the current SVID and bundle as an :class:`MTLSConfig`.

        Returns:
            MTLSConfig: Certificate chain, key, and the trust-domain bundle;
            client certificates are required and limited to
            :attr:`SPIFFEConfig.allowed_ids`.

        Raises:
            RuntimeError: When the SPIFFE SDK is missing or the Workload API
                returns no usable SVID.
        """
        try:
            from spiffe import WorkloadApiClient
        except ImportError as exc:
            raise RuntimeError("SPIFFE needs the 'spiffe' package; install membrane[tls-spiffe]") from exc
        with WorkloadApiClient(socket_path=self.config.socket_path) as client:
            context = client.fetch_x509_context()
        svid = context.default_svid
        if svid is None or not svid.cert_chain:
            raise RuntimeError("the SPIFFE Workload API returned no X.509 SVID")
        bundle = context.x509_bundle_set.get_bundle_for_trust_domain(svid.spiffe_id.trust_domain)
        authorities = list(bundle.x509_authorities) if bundle is not None else []
        if not authorities:
            raise RuntimeError(f"no trust bundle for {svid.spiffe_id.trust_domain}")
        chain = "".join(to_pem(cert) for cert in svid.cert_chain)
        key = svid.private_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ).decode()
        logger.info("fetched SVID %s", svid.spiffe_id)
        return MTLSConfig(
            server_cert_pem=chain,
            server_key_pem=key,
            ca_bundle_pem="".join(to_pem(cert) for cert in authorities),
            allowed_cns=self.config.allowed_ids,
            client_cert_pem=chain,
            client_key_pem=key,
            verify_hostname=False,
        )


__all__ = ["REFRESH_INTERVAL_SEC", "SPIFFEClient", "SPIFFEConfig", "to_pem"]
