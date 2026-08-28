"""Tests for observability endpoints (/livez, /readyz, /metrics)."""

import pytest
from fastapi.testclient import TestClient

from membrane.metrics import MetricsCollector
from membrane.node import Node
from membrane.transport.fastapi import create_app
from membrane.transfer import TransferService


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


def test_readyz_returns_503_when_over_capacity(client):
    """``GET /readyz`` returns 503 when the node has exhausted its memory budget."""
    c, registry = client
    from membrane.fragment import Fragment
    from membrane.signature import Signature

    # Force the node to a 1-byte budget and store a fragment larger than it.
    c.app.state.node.max_memory_bytes = 1  # type: ignore[attr-defined]
    sig = Signature(model_id="m", layer_range=(0, 1), token_span=(0, 1))
    big = Fragment(
        content_hash="h",
        embedding=(0.0,),
        structural_signature=sig,
        size=1024,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )
    # Store via direct dict mutation so we don't trigger the store logic's checks.
    c.app.state.node.fragments["h"] = big  # type: ignore[attr-defined]
    # Inject the equivalent of get_stats() reporting full memory.
    from dataclasses import replace
    c.app.state.node.get_stats = lambda: type("S", (), {  # type: ignore[attr-defined]
        "memory_used_bytes": 1,
        "memory_limit_bytes": 1,
        "fragment_count": 1,
        "primary_count": 0,
    })()
    resp = c.get("/readyz")
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
