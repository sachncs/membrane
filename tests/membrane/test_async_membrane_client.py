"""Tests for the async MembraneClient (Phase 3.6.1 follow-up)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from membrane.client import (
    AsyncMembraneClient,
    MembraneNotFoundError,
    MembraneServerError,
)


def _make_async_transport(routes: dict[tuple[str, str], tuple[int, dict | str]]):
    """Build an :class:`httpx.MockTransport` for the async client.

    Args:
        routes: ``{(method, path): (status, body)}``.

    Returns:
        httpx.MockTransport: Bound handler.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "no route"})
        status, body = routes[key]
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)
    return httpx.MockTransport(handler)


class TestAsyncMembraneClient:
    def test_store_returns_dict(self):
        routes = {("POST", "/store"): (200, {"success": True, "content_hash": "h"})}
        transport = _make_async_transport(routes)

        async def run() -> dict:
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                client = AsyncMembraneClient("http://t", client=c)
                return await client.store(
                    {
                        "schema_version": 5,
                        "tenant_id": "public",
                        "identity": {},
                        "payload_size": 0,
                        "ttl": 0,
                        "reuse_score": 0,
                        "version_id": 1,
                    }
                )

        assert asyncio.run(run()) == {"success": True, "content_hash": "h"}

    def test_retrieve_404_raises(self):
        routes = {("GET", "/retrieve"): (404, {"error": "missing"})}
        transport = _make_async_transport(routes)

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                client = AsyncMembraneClient("http://t", client=c)
                return await client.retrieve("missing")

        with pytest.raises(MembraneNotFoundError):
            asyncio.run(run())

    def test_retrieve_500_raises_server_error(self):
        routes = {("GET", "/retrieve"): (503, {"error": "draining"})}
        transport = _make_async_transport(routes)

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                client = AsyncMembraneClient("http://t", client=c)
                return await client.retrieve("missing")

        with pytest.raises(MembraneServerError):
            asyncio.run(run())

    def test_inventory_200(self):
        routes = {("GET", "/inventory"): (200, {"node_id": "n1", "digest": {}})}
        transport = _make_async_transport(routes)

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                client = AsyncMembraneClient("http://t", client=c)
                return await client.inventory()

        assert asyncio.run(run()) == {"node_id": "n1", "digest": {}}
