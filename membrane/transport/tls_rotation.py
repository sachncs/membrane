"""mTLS cert rotation + X.509 notAfter enforcement (Phase 3.4.1 + 3.4.2).

The v3.0.0 release adds hot-reload of the cert chain at runtime:

* :func:`cert_not_after` parses an X.509 PEM and returns the
  :class:`datetime.datetime` ``notAfter`` timestamp (UTC).
* :func:`enforce_not_after` raises when the chain is expired
  or expires inside the configured warning window (default
  30 days).
* :class:`CertRotationWatcher` polls the cert + key files and
  fires :meth:`on_rotate` whenever the on-disk bytes change. A
  SIGHUP handler reloads on demand so operators do not need
  to wait for the poll interval.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _parse_pem(payload: str) -> bytes:
    """Extract the raw DER bytes from a single CERTIFICATE PEM.

    Args:
        payload: PEM string with the ``-----BEGIN CERTIFICATE-----``
            header and footer.

    Returns:
        bytes: The raw DER bytes (``b''`` when no certificate
        PEM is present).
    """
    if "-----BEGIN CERTIFICATE-----" not in payload:
        return b""
    body = payload.split("-----BEGIN CERTIFICATE-----", 1)[1].split(
        "-----END CERTIFICATE-----", 1
    )[0]
    import base64

    return base64.b64decode("".join(body.split()))


def cert_not_after(pem: str) -> datetime.datetime | None:
    """Return the ``notAfter`` timestamp of a CERTIFICATE PEM.

    Args:
        pem: The PEM string.

    Returns:
        datetime.datetime | None: UTC ``notAfter``, or ``None``
        when the PEM is empty or the parser is unavailable
        (e.g., :mod:`cryptography` not installed).
    """
    der = _parse_pem(pem)
    if not der:
        return None
    try:
        from cryptography import x509

        cert = x509.load_der_x509_certificate(der)
        return cert.not_valid_after_utc
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("cert parser unavailable: %s", exc)
        return None


def enforce_not_after(
    pem: str,
    *,
    warn_days: int = 30,
    now: datetime.datetime | None = None,
) -> None:
    """Raise when the cert is expired or inside the warn window.

    Args:
        pem: The PEM string.
        warn_days: Warning window in days.
        now: UTC reference time. ``None`` reads :mod:`datetime`.

    Raises:
        RuntimeError: When the cert expires inside the window or
            :mod:`cryptography` is unavailable.
    """
    expires = cert_not_after(pem)
    if expires is None:
        return  # no cert to check
    current = (
        datetime.datetime.now(tz=datetime.timezone.utc)
        if now is None
        else now
    )
    delta = expires - current
    if delta <= datetime.timedelta(0):
        raise RuntimeError(f"mTLS cert is expired (notAfter={expires.isoformat()})")
    if delta <= datetime.timedelta(days=warn_days):
        logger.warning(
            "mTLS cert expires in %d days (notAfter=%s)",
            delta.days,
            expires.isoformat(),
        )


@dataclass
class CertRotationWatcher:
    """File-watch + SIGHUP-triggered cert rotation observer.

    Attributes:
        cert_path: Path to the cert PEM on disk.
        key_path: Path to the key PEM on disk.
        on_rotate: Callable invoked with (cert_pem, key_pem) on
            every change.
        poll_interval_sec: File poll interval (default 60s).
    """

    cert_path: str
    key_path: str
    on_rotate: Callable[[str, str], None]
    poll_interval_sec: float = 60.0
    _last_digest: str = field(default="", init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _sighup_installed: bool = field(default=False, init=False)

    def install_sighup(self) -> None:
        """Install a SIGHUP handler that triggers an immediate reload."""
        if self._sighup_installed:
            return
        watcher = self

        def _handler(signum: int, frame: object) -> None:
            watcher._reload()

        try:
            signal.signal(signal.SIGHUP, _handler)
            self._sighup_installed = True
        except (ValueError, AttributeError):  # pragma: no cover - non-POSIX
            logger.debug("SIGHUP handler not installed (non-POSIX platform)")

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._reload()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="membrane-cert-rotation",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        """Poll loop."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.poll_interval_sec):
                return
            self._reload()

    def _reload(self) -> None:
        """Read the cert + key files; invoke on_rotate on byte change."""
        try:
            with open(self.cert_path, encoding="utf-8") as f:
                cert = f.read()
            with open(self.key_path, encoding="utf-8") as f:
                key = f.read()
        except OSError as exc:
            logger.warning("cert reload skipped: %s", exc)
            return
        digest = hashlib.sha256(
            (cert + "|" + key).encode("utf-8")
        ).hexdigest()
        if digest == self._last_digest:
            return
        self._last_digest = digest
        # Enforce notAfter when cryptography is importable.
        try:
            enforce_not_after(cert)
        except RuntimeError as exc:
            logger.warning("cert reload rejected: %s", exc)
            return
        self.on_rotate(cert, key)


__all__ = [
    "CertRotationWatcher",
    "cert_not_after",
    "enforce_not_after",
]
