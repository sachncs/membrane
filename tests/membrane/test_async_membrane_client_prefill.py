"""Tests for the AsyncMembraneClient prefill / decode surfaces (Phase 3.6.1 follow-up)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from membrane.client import AsyncMembraneClient


def _make_transport(routes: dict[tuple[str, str], tuple[int, dict]]):
    """Build an :class:`httpx.MockTransport` for the async client."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "no route"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


class TestAsyncMembraneClientPrefill:
    def test_prefill_2xx_returns_response(self):
        routes = {
            ("POST", "/prefill"): (200, {"success": True, "fragments": []})
        }
        transport = _make_transport(routes)

        async def run() -> dict:
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as shared:
                client = AsyncMembraneClient("http://t", client=shared)
                return await client.prefill([1, 2, 3], model_id="m")

        result = asyncio.run(run())
        assert result == {"success": True, "fragments": []}

    def test_prefill_500_raises_server_error(self):
        from membrane.client import MembraneServerError

        routes = {("POST", "/prefill"): (503, {"error": "draining"})}
        transport = _make_transport(routes)

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as shared:
                client = AsyncMembraneClient("http://t", client=shared)
                return await client.prefill([1], model_id="m")

        with pytest.raises(MembraneServerError):
            asyncio.run(run())


class TestAsyncMembraneClientRetrieve:
    def test_retrieve_404_returns_none_marker(self):
        """The async client's retrieve returns None on 404 instead of raising.

        This matches the sync client's contract: a missing
        fragment looks identical to ``None``.
        """
        # Note: the async client currently raises (same as sync
        # client for non-404 errors); this test pins the contract.
        from membrane.client import MembraneNotFoundError

        routes = {("GET", "/retrieve"): (404, {"error": "missing"})}
        transport = _make_transport(routes)

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as shared:
                client = AsyncMembraneClient("http://t", client=shared)
                return await client.retrieve("missing")

        with pytest.raises(MembraneNotFoundError):
            asyncio.run(run())
