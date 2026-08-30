"""Outbound URL allow-list (Phase 3.1.2).

The v2.0 release shipped
:func:`membrane.transport.ops.op_sync` and the
:class:`membrane.network.peer.HTTPTransport` calling
``urllib.request.urlopen`` on caller-supplied URLs with no
scheme, host, or DNS-rebinding protection. A malicious
``POST /sync`` body could point at ``file:///etc/passwd``,
``http://169.254.169.254/...`` (cloud metadata), or
``http://10.0.0.1/admin`` (cluster-internal service).

The v3.0.0 release routes every outbound URL through
:func:`validate_outbound_url`. The check:

* restricts the scheme to ``http`` or ``https``;
* blocks RFC 1918 private addresses, the link-local
  ``169.254.0.0/16`` block (cloud metadata services),
  ``127.0.0.0/8`` (loopback), ``::1`` (IPv6 loopback), and
  the IPv6 ULA ``fc00::/7``;
* resolves the hostname once and compares the resolved
  address against the same blocklist (DNS-rebinding guard);
* honors a configured host allow-list as an escape hatch for
  deployments that need to talk to a specific metadata
  service or a CI-only external URL.

Operators that need to allow a private address (e.g., a
sidecar metadata service reachable on a private network) add
the exact host string to :class:`URLAllowlist.allowlist`. The
allowlist is consulted before the blocklist.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SSRFError(ValueError):
    """Raised when an outbound URL fails the allow-list check."""


_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """Return True if ``ip`` falls in a blocked private range.

    Args:
        ip: The resolved address to test.

    Returns:
        bool: True when the address is private, loopback,
        link-local, or otherwise not routable from the cluster
        edge.
    """
    private = getattr(ip, "is_private", False)
    loopback = getattr(ip, "is_loopback", False)
    link_local = getattr(ip, "is_link_local", False)
    multicast = getattr(ip, "is_multicast", False)
    unspecified = getattr(ip, "is_unspecified", False)
    reserved = getattr(ip, "is_reserved", False)
    return bool(
        private or loopback or link_local or multicast or unspecified or reserved
    )


@dataclass(frozen=True)
class URLAllowlist:
    """Outbound URL policy.

    Attributes:
        allowlist: Exact hostnames that bypass the blocklist. Use
            for deployments that intentionally talk to a
            private-network service (e.g., a co-located
            metadata service or a CI mirror).
        block_private: When ``True`` (the default), resolve the
            hostname and reject any address that lands in a
            private / loopback / link-local range. Set to
            ``False`` to skip the DNS check (e.g., tests that
            only exercise the scheme check).
        resolve_timeout: Seconds to spend on the DNS resolve.
    """

    allowlist: frozenset[str] = field(default_factory=frozenset)
    block_private: bool = True
    resolve_timeout: float = 2.0

    def is_host_allowed(self, hostname: str) -> bool:
        """Return True if ``hostname`` is on the explicit allow-list.

        Args:
            hostname: Hostname (lowercased) to look up.

        Returns:
            bool: True when the hostname is in the allow-list.
        """
        return hostname.lower() in self.allowlist


_DEFAULT_ALLOWLIST: URLAllowlist = URLAllowlist()


def get_default_allowlist() -> URLAllowlist:
    """Return the process-wide default :class:`URLAllowlist`.

    Operators that need a custom allow-list configure
    :func:`set_default_allowlist` at startup. Tests can call
    :func:`set_default_allowlist` with a relaxed policy and
    restore it via :func:`reset_default_allowlist` to keep
    state out of the test global.

    Returns:
        URLAllowlist: The current default.
    """
    return _DEFAULT_ALLOWLIST


def set_default_allowlist(allowlist: URLAllowlist) -> None:
    """Replace the process-wide default :class:`URLAllowlist`.

    Args:
        allowlist: The new default.
    """
    global _DEFAULT_ALLOWLIST
    _DEFAULT_ALLOWLIST = allowlist


def reset_default_allowlist() -> None:
    """Restore the default :class:`URLAllowlist` to its factory state.

    Tests use this to undo a :func:`set_default_allowlist`
    call without leaking policy into other tests.
    """
    global _DEFAULT_ALLOWLIST
    _DEFAULT_ALLOWLIST = URLAllowlist()


def _resolve_addresses(hostname: str) -> list[ipaddress._BaseAddress]:
    """Resolve ``hostname`` and return every IP it points at.

    Args:
        hostname: The hostname to resolve.

    Returns:
        List of IPv4 / IPv6 addresses.

    Raises:
        socket.gaierror: When DNS returns no records.
    """
    infos = socket.getaddrinfo(hostname, None)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def validate_outbound_url(
    url: str,
    allowlist: URLAllowlist | None = None,
) -> str:
    """Validate ``url`` against the SSRF policy.

    Args:
        url: The outbound URL to validate.
        allowlist: Optional :class:`URLAllowlist` to apply.
            Defaults to the process-wide policy from
            :func:`get_default_allowlist`.

    Returns:
        str: The validated URL (unchanged on success).

    Raises:
        SSRFError: When the URL is malformed, has a disallowed
            scheme, points at a blocked host, or resolves to a
            blocked IP range.
    """
    policy = allowlist or get_default_allowlist()
    if not isinstance(url, str) or not url:
        raise SSRFError("empty url")
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise SSRFError(f"malformed url: {exc}") from exc
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise SSRFError(f"scheme not allowed: {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise SSRFError("missing host")
    if policy.is_host_allowed(host):
        return url
    if not policy.block_private:
        return url
    try:
        addresses = _resolve_addresses(host)
    except socket.gaierror as exc:
        raise SSRFError(f"dns resolution failed for {host!r}: {exc}") from exc
    if not addresses:
        raise SSRFError(f"no addresses for {host!r}")
    for ip in addresses:
        if _is_blocked_ip(ip):
            raise SSRFError(f"host {host!r} resolves to blocked address {ip}")
    return url


def configure(
    allowlist: Iterable[str] = (),
    block_private: bool = True,
) -> URLAllowlist:
    """Configure the process-wide default :class:`URLAllowlist`.

    Args:
        allowlist: Iterable of hostnames to add to the
            allow-list.
        block_private: Whether to enforce the private-IP
            blocklist.

    Returns:
        URLAllowlist: The newly installed default.
    """
    new = URLAllowlist(
        allowlist=frozenset(h.lower() for h in allowlist),
        block_private=block_private,
    )
    set_default_allowlist(new)
    return new


__all__ = [
    "SSRFError",
    "URLAllowlist",
    "configure",
    "get_default_allowlist",
    "reset_default_allowlist",
    "set_default_allowlist",
    "validate_outbound_url",
]
