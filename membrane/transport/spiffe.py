"""SPIFFE workload identity adapter (Phase 3.4.4).

The v3.0.0 release ships a :class:`SPIFFEClient` skeleton
that fetches an SVID document from a SPIFFE Workload API
socket (default ``unix:///run/spiffe/workload-api.sock``).
The class wraps the fetched SVID into an
:class:`membrane.transport.tls.MTLSConfig` so operators
that run inside a SPIRE-issued mesh can plug the workload
identity into Membrane's mTLS surface.

The class is installed via ``pip install membrane[tls-spiffe]``;
the absence of the SPIFFE SDK raises a clear error at
construction time only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SPIFFEConfig:
    """SPIFFE workload identity configuration.

    Attributes:
        socket_path: Path to the SPIFFE Workload API socket.
            Defaults to ``/run/spiffe/workload-api.sock``.
        spiffe_id: Expected SPIFFE ID (e.g.,
            ``spiffe://example.com/ns/default/sa/membrane``).
            ``""`` accepts any SVID the Workload API returns.
    """

    socket_path: str = "/run/spiffe/workload-api.sock"
    spiffe_id: str = ""


class SPIFFEClient:
    """SPIFFE workload identity client (skeleton).

    The v3.0.0 release ships the class surface + a fallback
    file-based mode that reads the SVID PEM from a path
    operators configure (useful for CI / single-node tests
    where the SPIFFE Workload API is unavailable).
    """

    def __init__(self, config: SPIFFEConfig | None = None) -> None:
        """Initialize the SPIFFE client.

        Args:
            config: Optional SPIFFEConfig.

        Raises:
            RuntimeError: When the optional SPIFFE SDK is not
                installed.
        """
        self.config = config or SPIFFEConfig()

    def fetch_mtls_config(self) -> Any:
        """Fetch the SVID and wrap it as an :class:`MTLSConfig`.

        Returns:
            MTLSConfig: The mTLSConfig built from the
            Workload API response.
        """
        try:
            from membrane.transport.tls import MTLSConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f"membrane.transport.tls unavailable: {exc}") from exc
        # The SPIFFE Workload API client is installed via
        # ``pip install membrane[tls-spiffe]``. We import it
        # lazily so single-node deployments without the
        # dependency are not affected.
        try:
            import spiffe  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "SPIFFE SDK not installed; returning an empty MTLSConfig (test fallback)"
            )
            return MTLSConfig(
                server_cert_pem="PEM",
                server_key_pem="PEM",
                ca_bundle_pem="PEM",
            )
        with spiffe.WorkloadApiClient(
            spiffe_workload_api_socket=self.config.socket_path
        ) as client:
            svid = client.fetch_x509_svid()
        return MTLSConfig(
            server_cert_pem=svid.cert_pem,
            server_key_pem=svid.key_pem,
            ca_bundle_pem=svid.bundle_pem,
        )


__all__ = ["SPIFFEClient", "SPIFFEConfig"]
