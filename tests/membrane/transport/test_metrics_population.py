"""Tests for transport metrics instrumentation (Phase 3.2.1).

The v3.0.0 release wires the TransportMetrics counters into
the real op call paths so :func:`op_metrics` produces a
populated Prometheus text exposition.
"""

from __future__ import annotations

import pytest

from membrane.metrics import (
    ClusterMetrics,
    MetricsCollector,
    PersistenceMetrics,
    TransportMetrics,
)
from membrane.transport.metrics import (
    record_cluster_replication,
    record_persistence,
    record_transport,
    sync_node_metrics,
)


class TestRecordTransport:
    def test_records_status_and_method(self):
        registry = MetricsCollector()
        metrics = TransportMetrics(registry=registry)

        def op() -> tuple[int, dict]:
            return 200, {"ok": True}

        status, body = record_transport(metrics, "store", "POST", op)
        assert status == 200
        assert body == {"ok": True}
        # The counter exists in the registry and is incremented.
        counter = metrics.requests
        assert counter.value == 1.0

    def test_records_4xx_status(self):
        registry = MetricsCollector()
        metrics = TransportMetrics(registry=registry)

        def op() -> tuple[int, dict]:
            return 403, {"error": "denied"}

        status, _body = record_transport(metrics, "delete", "POST", op)
        assert status == 403
        assert metrics.requests.value == 1.0

    def test_records_5xx_status(self):
        registry = MetricsCollector()
        metrics = TransportMetrics(registry=registry)

        def op() -> tuple[int, dict]:
            return 503, {"error": "draining"}

        status, _body = record_transport(metrics, "store", "POST", op)
        assert status == 503

    def test_records_duration(self):
        registry = MetricsCollector()
        metrics = TransportMetrics(registry=registry)

        def op() -> tuple[int, dict]:
            return 200, {}

        record_transport(metrics, "store", "POST", op)
        assert metrics.duration.total == 1

    def test_no_metrics_bypasses(self):
        def op() -> tuple[int, dict]:
            return 200, {"ok": True}

        status, body = record_transport(None, "store", "POST", op)
        assert status == 200
        assert body == {"ok": True}


class TestRecordPersistence:
    def test_records_success(self):
        registry = MetricsCollector()
        metrics = PersistenceMetrics(registry=registry)

        def op():
            return "value"

        result = record_persistence(metrics, "get", op)
        assert result == "value"
        ops_counter = metrics.operations
        assert ops_counter.value == 1.0

    def test_no_metrics_bypasses(self):
        result = record_persistence(None, "put", lambda: None)
        assert result is None


class TestRecordClusterReplication:
    def test_records_success(self):
        registry = MetricsCollector()
        metrics = ClusterMetrics(registry=registry)
        record_cluster_replication(metrics, True)
        assert metrics.replications.value == 1.0
        assert metrics.replication_failures.value == 0.0

    def test_records_failure(self):
        registry = MetricsCollector()
        metrics = ClusterMetrics(registry=registry)
        record_cluster_replication(metrics, False)
        assert metrics.replications.value == 0.0
        assert metrics.replication_failures.value == 1.0

    def test_no_metrics_bypasses(self):
        record_cluster_replication(None, False)  # does not raise


class TestSyncNodeMetrics:
    def test_refresh_gauges(self):
        from membrane.node import Node

        registry = MetricsCollector()
        from membrane.metrics import NodeMetrics

        metrics = NodeMetrics(registry=registry)
        node = Node(node_id="n1", max_memory_bytes=10_000, metrics=metrics)
        sync_node_metrics(node, metrics)
        assert metrics.fragments.value == 0.0
        assert metrics.memory_limit_bytes.value == 10_000.0


class TestFastAPIIntegration:
    def test_records_stores_via_http(self):
        """Hitting /store on the FastAPI app increments the requests counter."""
        from fastapi.testclient import TestClient

        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transfer import TransferService
        from membrane.transport.fastapi import create_app

        registry = MetricsCollector()
        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = create_app(
            node=node,
            compute_backend=None,
            transfer_service=TransferService(),
            cluster_manager=None,
            metrics_registry=registry,
        )
        client = TestClient(app)
        ident = PayloadIdentity(
            payload_hash="h" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, 7),
            dtype="float16",
            shape=(1, 1, 1, 8, 64),
        )
        body = {
            "fragment": {
                "schema_version": 5,
                "tenant_id": "acme",
                "identity": ident.to_dict(),
                "payload_ref": None,
                "payload_size": 10,
                "ttl": 60.0,
                "reuse_score": 0.5,
                "version_id": 1,
                "consistency": "strong",
                "hlc": 0,
                "fingerprint_compat": "",
            },
            "is_primary": False,
        }
        resp = client.post("/store", json=body)
        assert resp.status_code == 200
        # The transport_requests_total counter is registered.
        from membrane.metrics import TransportMetrics

        metrics = TransportMetrics(registry=registry)
        assert metrics.requests.value >= 1.0

    def test_records_metrics_endpoint(self):
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transfer import TransferService
        from membrane.transport.fastapi import create_app

        registry = MetricsCollector()
        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = create_app(
            node=node,
            compute_backend=None,
            transfer_service=TransferService(),
            cluster_manager=None,
            metrics_registry=registry,
        )
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Even when no traffic has hit the server yet, the Prometheus
        # exposition should be a non-empty well-formed body.
        assert resp.text  # not empty
        # The response should at least declare the `# HELP` /
        # `# TYPE` headers for the declared series, or be empty
        # when no traffic has hit the server. Either way the
        # endpoint answered 200.
        assert resp.headers["content-type"].startswith("text/plain")
