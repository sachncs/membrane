"""Peer: HTTP client for inter-node communication.

Speaks the same REST surface as
:class:`~membrane.transport.fastapi.FastAPIServer`, exposing methods for
the cluster-management verbs (``join``, ``leave``, ``heartbeat``,
``gossip``) and the fragment-management verbs (``store``,
``retrieve``, ``replicate``).

The wire-level HTTP work is delegated to a pluggable
:class:`Transport` (default: :class:`HTTPTransport`). The default
transport uses ``urllib.request``; tests use
``unittest.mock.patch`` on the transport instance directly.

Thread safety:
    The class is **not** explicitly thread-safe; in practice a
    client is bound to a single peer and shared across the
    background threads that talk to that peer. The default
    :class:`HTTPTransport` handles concurrent sockets internally.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol, runtime_checkable

from membrane.fragment import Fragment
from membrane.serialization import from_dict, to_dict

logger = logging.getLogger(__name__)


#: A JSON object (used for every Membrane wire payload).
JsonDict = dict[str, Any]


@runtime_checkable
class Transport(Protocol):
    """Pluggable wire-level HTTP transport.

    Implementations must return a ``dict`` (parsed JSON body) on a
    successful 2xx response and ``None`` on any non-retryable
    failure. Retry semantics are the caller's responsibility; this
    protocol is intentionally minimal.
    """

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> JsonDict | None:
        """Issue one HTTP request and return the parsed JSON body.

        Args:
            method: HTTP method.
            url: Full URL.
            body: Request body bytes or ``None``.
            headers: Request headers.
            timeout_sec: Per-request timeout in seconds.

        Returns:
            JsonDict | None: Parsed JSON body on success,
            ``None`` on non-retryable failure.
        """
        ...


class HTTPTransport:
    """Default :class:`Transport` backed by ``urllib.request``."""

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> JsonDict | None:
        """Issue an HTTP request via ``urllib`` and return the JSON body."""
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError:
            return None
        except Exception as exc:
            logger.debug("HTTPTransport %s %s failed: %s", method, url, exc)
            return None


class Peer:
    """HTTP client for a single Membrane peer.

    Args:
        base_url: Peer URL (e.g., ``http://192.168.1.2:8080``).
        transport: :class:`Transport` instance. Defaults to a
            shared :class:`HTTPTransport`.
        timeout_sec: Request timeout.
        max_retries: Max retry attempts.
        retry_delay_sec: Base delay between retries.
    """

    def __init__(
        self,
        base_url: str,
        transport: Transport | None = None,
        timeout_sec: float = 5.0,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
        local_peer_cn: str = "",
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Peer URL. Trailing slashes are stripped.
            transport: Optional :class:`Transport`; defaults to
                :class:`HTTPTransport`.
            timeout_sec: Per-request timeout in seconds.
            max_retries: Maximum number of attempts before
                giving up.
            retry_delay_sec: Base delay used as ``base *
                2 ** attempt`` for exponential backoff.
            local_peer_cn: Common Name this client presents as
                the verified peer on the wire. When non-empty,
                every outbound request carries an
                ``X-Local-Peer-CN`` header so the receiving
                :class:`~membrane.network.membership.Membership`
                records and validates the caller's identity at
                the membership layer.
        """
        self.base_url = base_url.rstrip("/")
        self.transport = transport or HTTPTransport()
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.local_peer_cn = local_peer_cn

    @property
    def base_headers(self) -> dict[str, str]:
        """Headers attached to every outbound request.

        Returns:
            dict[str, str]: ``User-Agent`` plus the optional
            ``X-Local-Peer-CN`` header. The receiving
            :class:`~membrane.transport.ops.op_heartbeat` reads
            the latter to update the membership's per-peer CN.
        """
        headers = {"User-Agent": "membrane-peer/2.0"}
        if self.local_peer_cn:
            headers["X-Local-Peer-CN"] = self.local_peer_cn
        return headers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heartbeat(self) -> JsonDict | None:
        """Send ``GET /heartbeat`` to the peer.

        Returns:
            JsonDict | None: Parsed JSON response, or ``None`` on
            failure.
        """
        return self.request_with_retry(
            "GET", "/heartbeat", extra_headers=self.base_headers
        )

    def get_inventory(self) -> JsonDict | None:
        """Send ``GET /inventory`` to the peer."""
        return self.request_with_retry("GET", "/inventory")

    def store_fragment(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Send ``POST /store`` with ``fragment`` and ``is_primary``."""
        payload = {"fragment": to_dict(fragment), "is_primary": is_primary}
        resp = self.request_with_retry("POST", "/store", payload)
        return resp is not None and resp.get("success", False)

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Send ``GET /retrieve?content_hash=...``."""
        resp = self.request_with_retry("GET", f"/retrieve?content_hash={content_hash}")
        if resp and resp.get("found"):
            return from_dict(resp["fragment"])
        return None

    def join_cluster(self, node_id: str, host: str, port: int) -> JsonDict | None:
        """Send ``POST /join`` to bootstrap into the cluster."""
        return self.request_with_retry("POST", "/join", {"node_id": node_id, "host": host, "port": port})

    def leave_cluster(self, node_id: str) -> bool:
        """Send ``POST /leave`` to remove ``node_id`` from the cluster."""
        resp = self.request_with_retry("POST", "/leave", {"node_id": node_id})
        return resp is not None and resp.get("success", False)

    def gossip(self, state: JsonDict) -> JsonDict | None:
        """Send ``POST /gossip`` with the supplied state payload."""
        return self.request_with_retry("POST", "/gossip", state)

    def request_replicate(self, fragment: Fragment) -> bool:
        """Send ``POST /replicate`` with ``fragment``."""
        payload = {"fragment": to_dict(fragment)}
        resp = self.request_with_retry("POST", "/replicate", payload)
        return resp is not None and resp.get("success", False)

    def request_delete(
        self,
        content_hash: str,
        node_id: str,
        tombstone_until: float | None = None,
    ) -> bool:
        """Send ``POST /delete`` for ``content_hash``.

        Args:
            content_hash: Hash to soft-delete.
            node_id: Local node identifier propagated to the peer.
            tombstone_until: Optional Unix deadline.

        Returns:
            bool: ``True`` when the peer's :func:`op_delete`
            succeeded.
        """
        body: dict[str, object] = {
            "content_hash": content_hash,
            "node_id": node_id,
        }
        if tombstone_until is not None:
            body["tombstone_until"] = tombstone_until
        resp = self.request_with_retry("POST", "/delete", body)
        return resp is not None and bool(resp.get("success", False))

    def request_tombstone(
        self,
        content_hash: str,
        until: float,
        node_id: str,
    ) -> bool:
        """Send ``POST /tombstone`` marking a soft-delete.

        Args:
            content_hash: Hash to tombstone.
            until: Wall-clock deadline.
            node_id: Originating node identifier.

        Returns:
            bool: ``True`` when the peer's :func:`op_tombstone`
            succeeded.
        """
        body = {"content_hash": content_hash, "until": until, "node_id": node_id}
        resp = self.request_with_retry("POST", "/tombstone", body)
        return resp is not None and bool(resp.get("success", False))

    def get_peers(self) -> JsonDict | None:
        """Send ``GET /peers``."""
        return self.request_with_retry("GET", "/peers")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def request_with_retry(
        self,
        method: str,
        path: str,
        payload: JsonDict | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> JsonDict | None:
        """Issue an HTTP request with retries and exponential backoff.

        A ``None`` response from the transport is retried up to
        ``max_retries`` times with ``retry_delay_sec * 2 ** attempt``
        seconds between attempts. Tests that want to simulate
        transient failures can return ``None`` from a mocked
        transport.

        Args:
            method: HTTP method.
            path: URL path appended to ``self.base_url``.
            payload: Optional JSON-serializable body.
            extra_headers: Optional additional headers merged
                with the per-request content-type. Used by
                :meth:`heartbeat` to attach the local
                ``X-Local-Peer-CN``.

        Returns:
            JsonDict | None: Parsed JSON response or ``None`` on
            terminal failure.
        """
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload else None
        headers = {"Content-Type": "application/json"} if payload else {}
        if extra_headers:
            headers.update(extra_headers)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                resp = self.transport.request(
                    method=method,
                    url=url,
                    body=data,
                    headers=headers,
                    timeout_sec=self.timeout_sec,
                )
                if resp is not None:
                    return resp
            except Exception as exc:
                last_error = exc

            # Exponential backoff: 1x, 2x, 4x, ...
            delay = self.retry_delay_sec * (2**attempt)
            logger.debug(
                "Request to %s%s failed (attempt %s/%s), retrying in %.1fs",
                self.base_url,
                path,
                attempt + 1,
                self.max_retries,
                delay,
            )
            time.sleep(delay)

        logger.warning(
            "Request to %s%s failed after %s retries: %s",
            self.base_url,
            path,
            self.max_retries,
            last_error,
        )
        return None


__all__ = ["HTTPTransport", "Peer", "Transport"]
