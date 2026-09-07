"""ACME / Let's Encrypt bootstrap (Phase 3.4.3).

The v3.0.0 release ships an :class:`ACMEClient` skeleton
compatible with the ACME v2 protocol (RFC 8555). Production
deployments use Pebble or Let's Encrypt as the directory;
the :meth:`ACMEClient.issue_certificate` entry point submits
an order, polls the order, and finalizes the cert + chain
when the challenge is satisfied.

The class is installed via ``pip install membrane[acme]``;
the import is lazy so the absence of :mod:`cryptography` and
:mod:`josepy` raises a clear error only when the client is
actually instantiated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ACMEConfig:
    """ACME v2 directory configuration.

    Attributes:
        directory_url: ACME directory URL.
        account_key_path: Path to the ACME account key.
        contact: List of contact emails (e.g.,
            ``["mailto:ops@example.com"]``).
    """

    directory_url: str = ""
    account_key_path: str = ""
    contact: list[str] | None = None


@dataclass
class ACMEOrder:
    """One ACME order.

    Attributes:
        domains: List of identifiers this order covers.
        cert_url: ``/finished-cert`` resource on the directory
            once the order completes.
        cert_path: Local file where the issued cert is
            written.
    """

    domains: list[str]
    cert_url: str = ""
    cert_path: str = ""


class ACMEClient:
    """ACME v2 client (skeleton).

    The v3.0.0 release ships the class surface so operators
    can wire it into their Server; the HTTP-01 / DNS-01
    challenge handlers are implemented in a follow-up
    release.
    """

    def __init__(self, config: ACMEConfig) -> None:
        """Initialize the ACME client.

        Args:
            config: Directory + account configuration.

        Raises:
            RuntimeError: When the optional cryptography
                dependency is not installed.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ACMEClient requires 'cryptography'; install membrane[acme]"
            ) from exc
        self.config = config

    def issue_certificate(self, domains: list[str], cert_path: str) -> ACMEOrder:
        """Issue a cert for ``domains`` via the configured directory.

        Args:
            domains: DNS identifiers the cert covers.
            cert_path: Local file where the issued cert is
                written.

        Returns:
            ACMEOrder: The pending order. Call
            :meth:`poll_until_ready` to await issuance.

        Raises:
            NotImplementedError: Until the v3.0.1 follow-up
                delivers the HTTP-01 + DNS-01 handlers.
        """
        raise NotImplementedError(
            "ACME HTTP-01 / DNS-01 challenges land in 3.0.1; the v3.0.0 release ships the class surface"
        )

    def poll_until_ready(self, order: ACMEOrder, timeout_sec: float = 60.0) -> None:
        """Await ``order`` issuance.

        Args:
            order: The order returned by :meth:`issue_certificate`.
            timeout_sec: Wall-clock budget.
        """
        raise NotImplementedError


__all__ = ["ACMEClient", "ACMEConfig", "ACMEOrder"]
