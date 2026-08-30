"""Tests for observability endpoints (/livez, /readyz, /metrics)."""

import pytest
from fastapi.testclient import TestClient

from membrane.metrics import MetricsCollector
from membrane.node import Node
from membrane.transfer import TransferService
from membrane.transport.fastapi import create_app


@pytest.fixture
def client():
    node = Node("n1")
    transfer = TransferService()
    registry = MetricsCollector()
    app = create_app(
        node=node,
        compute_backend=None,
        transfer_service=transfer,
        cluster_manager=None,
        metrics_registry=registry,
    )
    return TestClient(app), registry


def test_livez_returns_alive(client):
    """``GET /livez`` returns 200 with status=alive."""
    c, _ = client
    resp = c.get("/livez")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readyz_returns_ready(client):
    """``GET /readyz`` returns 200 with status=ready when node has capacity."""
    c, _ = client
    resp = c.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_readyz_returns_503_when_over_capacity():
    """``GET /readyz`` returns 503 when the node's memory budget is exhausted."""
    from tests.conftest import make_fragment

    node = Node("n1", max_memory_bytes=10)
    transfer = TransferService()
    registry = MetricsCollector()
    app = create_app(
        node=node,
        compute_backend=None,
        transfer_service=transfer,
        cluster_manager=None,
        metrics_registry=registry,
    )
    # Saturate the node by stuffing its memory-usage counter
    # past the configured limit. The store() path requires
    # evict() to make room; bypassing it with a direct
    # fragment insertion exercises the readiness check
    # without engaging the eviction logic.
    node.fragments["h"] = make_fragment("h", size=11)
    node.memory_usage = node.max_memory_bytes
    test_client = TestClient(app)
    resp = test_client.get("/readyz")
    assert resp.status_code == 503


def test_metrics_returns_prometheus_text(client):
    """``GET /metrics`` returns Prometheus text exposition."""
    c, registry = client
    registry.counter("test_counter", "Test counter.").inc()
    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "# HELP test_counter" in body
    assert "# TYPE test_counter counter" in body
    assert "test_counter 1.0" in body


def test_metrics_json_returns_node_snapshot(client):
    """``GET /metrics.json`` returns the legacy JSON snapshot used by the TUI."""
    c, _ = client
    resp = c.get("/metrics.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "node_id" in data
    assert "memory_used_bytes" in data
    assert "fragment_count" in data
