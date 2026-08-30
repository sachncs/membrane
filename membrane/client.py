"""Typed Membrane client (Phase 3.6.1).

The v3.0.0 release ships a typed MembraneClient that wraps the
``/store``, ``/retrieve``, ``/prefill``, ``/decode``,
``/inventory``, ``/peers``, ``/heartbeat``, and ``/metrics``
endpoints in a single object. The synchronous variant uses
:mod:`httpx.Client`; the async variant uses
:class:`httpx.AsyncClient`.

Exceptions:

* :class:`MembraneClientError` -- base class.
* :class:`MembraneNotFoundError` -- 404.
* :class:`MembraneUnauthorizedError` -- 401 / 403.
* :class:`MembraneServerError` -- 5xx.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MembraneClientError(Exception):
    """Base class for Membrane HTTP client errors."""


class MembraneNotFoundError(MembraneClientError):
    """Raised on a 404."""


class MembraneUnauthorizedError(MembraneClientError):
    """Raised on a 401 / 403."""


class MembraneServerError(MembraneClientError):
    """Raised on a 5xx server-side failure."""


def _raise_for_status(status_code: int, body: Any) -> None:
    """Translate an HTTP status code into a typed exception.

    Args:
        status_code: The HTTP status code from the server.
        body: The decoded response body.

    Raises:
        MembraneNotFoundError: When the status is 404.
        MembraneUnauthorizedError: When the status is 401 / 403.
        MembraneServerError: When the status is 5xx.
    """
    if status_code == 404:
        raise MembraneNotFoundError(f"not found: {body!r}")
    if status_code in (401, 403):
        raise MembraneUnauthorizedError(f"unauthorized: {body!r}")
    if status_code >= 500:
        raise MembraneServerError(f"server error: {status_code} {body!r}")


class MembraneClient:
    """Synchronous typed client for the v3.0.0 HTTP API.

    Attributes:
        base_url: Server URL, e.g., ``http://node-1:8080``.
        api_key: Bearer token; ``""`` skips the Authorization
            header.
        timeout: Per-request timeout in seconds.
        transport: Optional :class:`httpx.Client` override.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout: float = 5.0,
        transport: httpx.Client | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Server URL.
            api_key: Bearer token (optional).
            timeout: Per-request timeout in seconds.
            transport: Optional transport override.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = transport or httpx.Client(timeout=timeout)

    @property
    def _headers(self) -> dict[str, str]:
        """Build the default request headers.

        Returns:
            dict: ``Authorization`` when ``api_key`` is set.
        """
        if self.api_key:
            return {"authorization": f"Bearer {self.api_key}"}
        return {}

    def store(self, fragment_payload: dict[str, Any], is_primary: bool = False) -> dict[str, Any]:
        """Call ``POST /store``.

        Args:
            fragment_payload: The :class:`FragmentPayload` body
                (``serialization.to_dict`` round-trip).
            is_primary: Whether the local node owns the primary
                shard.

        Returns:
            dict: The server's response body.

        Raises:
            MembraneClientError: subclass as appropriate.
        """
        resp = self._client.post(
            f"{self.base_url}/store",
            json={"fragment": fragment_payload, "is_primary": is_primary},
            headers=self._headers,
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def retrieve(self, content_hash: str) -> dict[str, Any] | None:
        """Call ``GET /retrieve?content_hash=...``.

        Args:
            content_hash: The hex digest of the content.

        Returns:
            dict: Server response; ``"found": False`` returns
            ``{"found": False, "fragment": None}``.
        """
        resp = self._client.get(
            f"{self.base_url}/retrieve",
            params={"content_hash": content_hash},
            headers=self._headers,
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def inventory(self) -> dict[str, Any]:
        """Call ``GET /inventory``.

        Returns:
            dict: ``{"node_id": ..., "digest": {...}}``.
        """
        resp = self._client.get(
            f"{self.base_url}/inventory", headers=self._headers
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def peers(self) -> dict[str, Any]:
        """Call ``GET /peers``."""
        resp = self._client.get(
            f"{self.base_url}/peers", headers=self._headers
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def heartbeat(self) -> dict[str, Any]:
        """Call ``GET /heartbeat``."""
        resp = self._client.get(
            f"{self.base_url}/heartbeat", headers=self._headers
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def prefill(self, prompt_tokens: list[int], model_id: str = "default") -> dict[str, Any]:
        """Call ``POST /prefill``."""
        resp = self._client.post(
            f"{self.base_url}/prefill",
            json={"prompt_tokens": prompt_tokens, "model_id": model_id},
            headers=self._headers,
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def metrics(self) -> str:
        """Call ``GET /metrics``.

        Returns:
            str: Prometheus text exposition.
        """
        resp = self._client.get(
            f"{self.base_url}/metrics", headers=self._headers
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.text

    def close(self) -> None:
        """Close the underlying :class:`httpx.Client`."""
        self._client.close()

    def __enter__(self) -> MembraneClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncMembraneClient:
    """Async typed client for the v3.0.0 HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the async client.

        Args:
            base_url: Server URL.
            api_key: Bearer token (optional).
            timeout: Per-request timeout.
            client: Optional :class:`httpx.AsyncClient` override.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def _headers(self) -> dict[str, str]:
        """Build the default request headers."""
        if self.api_key:
            return {"authorization": f"Bearer {self.api_key}"}
        return {}

    async def store(
        self, fragment_payload: dict[str, Any], is_primary: bool = False
    ) -> dict[str, Any]:
        resp = await self._client.post(
            f"{self.base_url}/store",
            json={"fragment": fragment_payload, "is_primary": is_primary},
            headers=self._headers,
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    async def retrieve(self, content_hash: str) -> dict[str, Any] | None:
        resp = await self._client.get(
            f"{self.base_url}/retrieve",
            params={"content_hash": content_hash},
            headers=self._headers,
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    async def inventory(self) -> dict[str, Any]:
        resp = await self._client.get(
            f"{self.base_url}/inventory", headers=self._headers
        )
        if resp.status_code >= 400:
            _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    async def close(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient`."""
        await self._client.aclose()


__all__ = [
    "AsyncMembraneClient",
    "MembraneClient",
    "MembraneClientError",
    "MembraneNotFoundError",
    "MembraneServerError",
    "MembraneUnauthorizedError",
]
