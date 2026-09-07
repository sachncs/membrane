"""Tests for the per-tenant cluster_metrics counter (Phase 3.1.7 follow-up)."""

from __future__ import annotations

import pytest

from membrane.metrics import ClusterMetrics, MetricsCollector
from membrane.transport.ops import op_store


def _store_payload(tenant: str = "acme") -> dict:
    """Build a v5 wire payload dict for the op_store call."""
    return {
        "schema_version": 5,
        "tenant_id": tenant,
        "identity": {
            "payload_hash": "h" * 64,
            "model_id": "m",
            "model_revision": "",
            "tokenizer_name": "m",
            "tokenizer_revision": "",
            "layer_range": (0, 1),
            "head_range": (-1, -1),
            "token_span": (0, 3),
            "dtype": "float16",
            "shape": (1, 1, 1, 1, 64),
        },
        "payload_ref": None,
        "payload_size": 10,
        "ttl": 60.0,
        "reuse_score": 0.5,
        "version_id": 1,
        "consistency": "strong",
        "hlc": 0,
        "fingerprint_compat": "",
    }


class TestOpStoreTenantMetrics:
    def test_successful_store_bumps_tenant_counter(self):
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=10_000)
        registry = MetricsCollector()
        cluster_metrics = ClusterMetrics(registry)

        status, body = op_store(
            node,
            _store_payload(tenant="acme"),
            cluster_metrics=cluster_metrics,
        )
        assert status == 200
        assert body["success"] is True
        # The per-tenant operation counter recorded the write.
        assert cluster_metrics.tenant.operation_count == {"acme": 1}

    def test_rejected_store_does_not_bump_counter(self):
        from membrane.auth import AuthContext
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=10_000)
        registry = MetricsCollector()
        cluster_metrics = ClusterMetrics(registry)

        # tenant_id on payload is "acme"; caller subject is
        # "globex" -- op_store returns 403 + no counter bump.
        context = AuthContext(subject="globex", scopes=frozenset({"write"}))
        status, _body = op_store(
            node,
            _store_payload(tenant="acme"),
            cluster_metrics=cluster_metrics,
            auth_context=context,
        )
        assert status == 403
        # No counter bump on a rejected store.
        assert cluster_metrics.tenant.operation_count == {}

    def test_no_metrics_when_unset(self):
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=10_000)
        # cluster_metrics=None should be a no-op.
        status, body = op_store(node, _store_payload(tenant="acme"))
        assert status == 200
        assert body["success"] is True
