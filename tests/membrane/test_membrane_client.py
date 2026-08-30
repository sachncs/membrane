"""Tests for the typed MembraneClient (Phase 3.6.1)."""

from __future__ import annotations

import json

import httpx
import pytest

from membrane.client import (
    AsyncMembraneClient,
    MembraneClient,
    MembraneClientError,
    MembraneNotFoundError,
    MembraneServerError,
    MembraneUnauthorizedError,
)


def _make_handler(routes: dict[tuple[str, str], tuple[int, dict | str]]) -> httpx.MockTransport:
    """Build an httpx MockTransport from a route -> (status, body) map.

    Args:
        routes: ``{(method, path): (status, body)}``.

    Returns:
        httpx.MockTransport: A bound handler.
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


class TestSyncMembraneClient:
    def test_store_2xx_returns_dict(self):
        routes = {("POST", "/store"): (200, {"success": True, "content_hash": "h"})}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client:
            result = client.store({"schema_version": 5, "tenant_id": "acme", "identity": {}, "payload_size": 0, "ttl": 60.0, "reuse_score": 0.5, "version_id": 1})
        assert result == {"success": True, "content_hash": "h"}

    def test_store_404_raises_not_found(self):
        routes = {("POST", "/store"): (404, {"error": "no_node"})}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client, pytest.raises(MembraneNotFoundError):
            client.store({"schema_version": 5})

    def test_store_403_raises_unauthorized(self):
        routes = {("POST", "/store"): (403, {"error": "no_scope"})}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client, pytest.raises(MembraneUnauthorizedError):
            client.store({"schema_version": 5})

    def test_store_500_raises_server_error(self):
        routes = {("POST", "/store"): (503, {"error": "draining"})}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client, pytest.raises(MembraneServerError):
            client.store({"schema_version": 5})

    def test_retrieve_200(self):
        routes = {("GET", "/retrieve"): (200, {"found": True, "fragment": None})}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client:
            result = client.retrieve("h" * 64)
        assert result == {"found": True, "fragment": None}

    def test_inventory_and_peers(self):
        routes = {
            ("GET", "/inventory"): (200, {"node_id": "n1", "digest": {"h": 1}}),
            ("GET", "/peers"): (200, {"peers": []}),
        }
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client:
            assert client.inventory()["node_id"] == "n1"
            assert client.peers() == {"peers": []}

    def test_metrics_returns_prometheus_text(self):
        routes = {("GET", "/metrics"): (200, "membrane_requests_total 0\n")}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client:
            assert "membrane_requests_total" in client.metrics()

    def test_prefill(self):
        routes = {("POST", "/prefill"): (200, {"success": True, "fragments": []})}
        transport = _make_handler(routes)
        with MembraneClient(
            "http://t", transport=httpx.Client(transport=transport, base_url="http://t")
        ) as client:
            assert client.prefill([1, 2, 3], model_id="m")["success"] is True


class TestErrorHierarchy:
    def test_subclass_relationships(self):
        assert issubclass(MembraneNotFoundError, MembraneClientError)
        assert issubclass(MembraneUnauthorizedError, MembraneClientError)
        assert issubclass(MembraneServerError, MembraneClientError)
