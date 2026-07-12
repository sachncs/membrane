"""PeerClient: HTTP client for inter-node communication.

Uses the standard library ``urllib.request`` so the network
layer has zero external dependencies. Requests carry a
configurable timeout, retry count, and exponential-backoff
delay.

The client is intentionally minimal — it speaks the same
REST surface as
:class:`~membrane.transport.http_server.HTTPServer`, exposing
methods for the cluster-management verbs (``join``, ``leave``,
``heartbeat``, ``gossip``) and the fragment-management verbs
(``store``, ``retrieve``, ``replicate``).

Thread safety:
    The class is **not** explicitly thread-safe; in practice a
    client is bound to a single peer and shared across the
    background threads that talk to that peer. ``urllib``
    handles concurrent sockets internally.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


def _serialize_fragment(frag: Fragment) -> dict[str, Any]:
    """Serialize a fragment to a JSON-compatible dict.

    Args:
        frag: Fragment to serialize.

    Returns:
        dict[str, Any]: Flat dict suitable for HTTP transport.
    """
    return {
        "content_hash": frag.content_hash,
        "embedding": list(frag.embedding),
        "model_id": frag.structural_signature.model_id,
        "layer_range": frag.structural_signature.layer_range,
        "token_span": frag.structural_signature.token_span,
        "size": frag.size,
        "ttl": frag.ttl,
        "reuse_score": frag.reuse_score,
        "version_id": frag.version_id,
    }


def _deserialize_fragment(data: dict[str, Any]) -> Fragment:
    """Reconstruct a fragment from its serialized form.

    Args:
        data: Mapping produced by :func:`_serialize_fragment`.

    Returns:
        Fragment: Reconstructed fragment instance.
    """
    return Fragment(
        content_hash=data["content_hash"],
        embedding=tuple(data["embedding"]),
        structural_signature=StructuralSignature(
            model_id=data["model_id"],
            layer_range=tuple(data["layer_range"]),
            token_span=tuple(data["token_span"]),
        ),
        size=data["size"],
        ttl=data["ttl"],
        reuse_score=data["reuse_score"],
        version_id=data["version_id"],
    )


class PeerClient:
    """HTTP client for a single Membrane peer.

    Args:
        base_url: Peer URL (e.g., ``http://192.168.1.2:8080``).
        timeout_sec: Request timeout.
        max_retries: Max retry attempts.
        retry_delay_sec: Base delay between retries.
    """

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 5.0,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Peer URL. Trailing slashes are stripped.
            timeout_sec: Per-request timeout in seconds.
            max_retries: Maximum number of attempts before
                giving up.
            retry_delay_sec: Base delay used as ``base *
                2 ** attempt`` for exponential backoff.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heartbeat(self) -> dict | None:
        """Send ``GET /heartbeat`` to the peer.

        Returns:
            dict | None: Parsed JSON response, or ``None`` on
            failure.
        """
        return self._request("GET", "/heartbeat")

    def get_inventory(self) -> dict | None:
        """Send ``GET /inventory`` to the peer.

        Returns:
            dict | None: Parsed JSON inventory response, or
            ``None`` on failure.
        """
        return self._request("GET", "/inventory")

    def store_fragment(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Send ``POST /store`` with ``fragment`` and ``is_primary``.

        Args:
            fragment: Fragment to store remotely.
            is_primary: Whether the remote node should claim
                primary ownership.

        Returns:
            bool: True when the peer confirmed the store,
            False otherwise (including network failures).
        """
        payload = {"fragment": _serialize_fragment(fragment), "is_primary": is_primary}
        resp = self._request("POST", "/store", payload)
        return resp is not None and resp.get("success", False)

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Send ``GET /retrieve?content_hash=...``.

        Args:
            content_hash: Hash of the fragment to retrieve.

        Returns:
            Fragment | None: The fragment, or ``None`` when
            the peer did not find it or the request failed.
        """
        resp = self._request("GET", f"/retrieve?content_hash={content_hash}")
        if resp and resp.get("found"):
            return _deserialize_fragment(resp["fragment"])
        return None

    def join_cluster(self, node_id: str, host: str, port: int) -> dict | None:
        """Send ``POST /join`` to bootstrap into the cluster.

        Args:
            node_id: Joining node's identifier.
            host: Joining node's host.
            port: Joining node's port.

        Returns:
            dict | None: Parsed JSON response (typically
            ``{"success": True, "peers": [...]}``) or ``None``
            on failure.
        """
        return self._request("POST", "/join", {"node_id": node_id, "host": host, "port": port})

    def leave_cluster(self, node_id: str) -> bool:
        """Send ``POST /leave`` to remove ``node_id`` from the cluster.

        Args:
            node_id: Leaving node's identifier.

        Returns:
            bool: True when the peer confirmed the leave,
            False otherwise.
        """
        resp = self._request("POST", "/leave", {"node_id": node_id})
        return resp is not None and resp.get("success", False)

    def gossip(self, state: dict) -> dict | None:
        """Send ``POST /gossip`` with the supplied state payload.

        Args:
            state: Pre-serialized gossip state.

        Returns:
            dict | None: Parsed JSON response or ``None`` on
            failure.
        """
        return self._request("POST", "/gossip", state)

    def request_replicate(self, fragment: Fragment) -> bool:
        """Send ``POST /replicate`` with ``fragment``.

        Args:
            fragment: Fragment to replicate on the peer.

        Returns:
            bool: True when the peer confirmed the replication,
            False otherwise.
        """
        payload = {"fragment": _serialize_fragment(fragment)}
        resp = self._request("POST", "/replicate", payload)
        return resp is not None and resp.get("success", False)

    def get_peers(self) -> dict | None:
        """Send ``GET /peers``.

        Returns:
            dict | None: Parsed membership payload or ``None``
            on failure.
        """
        return self._request("GET", "/peers")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict | None:
        """Issue an HTTP request with retries and exponential backoff.

        ``HTTP 400/404/503`` are treated as terminal failures
        (no retries). All other errors — connection refused,
        timeout, 5xx (other than 503), JSON decode errors — are
        retried up to ``max_retries`` times with
        ``retry_delay_sec * 2 ** attempt`` seconds between
        attempts.

        Args:
            method: HTTP method.
            path: URL path appended to ``self.base_url``.
            payload: Optional JSON-serializable body.

        Returns:
            dict | None: Parsed JSON response or ``None`` on
            terminal failure.
        """
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload else None
        headers = {"Content-Type": "application/json"} if payload else {}
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = resp.read().decode()
                    return json.loads(body) if body else {}
            except urllib.error.HTTPError as e:
                # 400/404/503 are non-retryable. 503 is included
                # because cluster peers use it to indicate "I am
                # shutting down" — retrying would just delay the
                # caller's failure path.
                if e.code in (404, 400, 503):
                    logger.debug(
                        "HTTP %s from %s%s: %s",
                        e.code, self.base_url, path, e.reason,
                    )
                    return None
                last_error = e
            except Exception as exc:
                last_error = exc

            # Exponential backoff: 1x, 2x, 4x, ...
            delay = self.retry_delay_sec * (2 ** attempt)
            logger.debug(
                "Request to %s%s failed (attempt %s/%s), retrying in %.1fs: %s",
                self.base_url, path, attempt + 1, self.max_retries, delay, last_error,
            )
            time.sleep(delay)

        logger.warning(
            "Request to %s%s failed after %s retries: %s",
            self.base_url, path, self.max_retries, last_error,
        )
        return None
