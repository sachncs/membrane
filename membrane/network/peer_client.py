"""PeerClient: HTTP client for inter-node communication.

Uses stdlib ``urllib.request`` with retry, timeout, and exponential backoff.
Zero external dependencies.
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
        base_url: Peer URL (e.g. ``http://192.168.1.2:8080``).
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
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heartbeat(self) -> dict | None:
        return self._request("GET", "/heartbeat")

    def get_inventory(self) -> dict | None:
        return self._request("GET", "/inventory")

    def store_fragment(self, fragment: Fragment, is_primary: bool = False) -> bool:
        payload = {"fragment": _serialize_fragment(fragment), "is_primary": is_primary}
        resp = self._request("POST", "/store", payload)
        return resp is not None and resp.get("success", False)

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        resp = self._request("GET", f"/retrieve?content_hash={content_hash}")
        if resp and resp.get("found"):
            return _deserialize_fragment(resp["fragment"])
        return None

    def join_cluster(self, node_id: str, host: str, port: int) -> dict | None:
        return self._request("POST", "/join", {"node_id": node_id, "host": host, "port": port})

    def leave_cluster(self, node_id: str) -> bool:
        resp = self._request("POST", "/leave", {"node_id": node_id})
        return resp is not None and resp.get("success", False)

    def gossip(self, state: dict) -> dict | None:
        return self._request("POST", "/gossip", state)

    def request_replicate(self, fragment: Fragment) -> bool:
        payload = {"fragment": _serialize_fragment(fragment)}
        resp = self._request("POST", "/replicate", payload)
        return resp is not None and resp.get("success", False)

    def get_peers(self) -> dict | None:
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
                if e.code in (404, 400, 503):
                    # Non-retryable
                    logger.debug("HTTP %s from %s%s: %s", e.code, self.base_url, path, e.reason)
                    return None
                last_error = e
            except Exception as exc:
                last_error = exc

            delay = self.retry_delay_sec * (2 ** attempt)
            logger.debug("Request to %s%s failed (attempt %s/%s), retrying in %.1fs: %s", self.base_url, path, attempt + 1, self.max_retries, delay, last_error)
            time.sleep(delay)

        logger.warning("Request to %s%s failed after %s retries: %s", self.base_url, path, self.max_retries, last_error)
        return None
