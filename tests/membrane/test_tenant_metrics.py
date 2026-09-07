"""Tests for per-tenant metrics (Phase 3.1.7)."""

from __future__ import annotations

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.metrics import (
    ClusterMetrics,
    MetricsCollector,
    NodeMetrics,
    TenantMetrics,
)


def _identity(payload_hash: str = "h" * 64) -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash=payload_hash,
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


class TestTenantMetrics:
    def test_defaults_to_empty(self):
        t = TenantMetrics()
        assert t.fragment_count == {}
        assert t.operation_count == {}

    def test_bump_fragment_creates_entry(self):
        t = TenantMetrics()
        t.bump_fragment("acme", 1)
        assert t.fragment_count == {"acme": 1}

    def test_bump_fragment_increments(self):
        t = TenantMetrics()
        t.bump_fragment("acme", 1)
        t.bump_fragment("acme", 2)
        assert t.fragment_count == {"acme": 3}

    def test_bump_fragment_negative_decrements(self):
        t = TenantMetrics()
        t.bump_fragment("acme", 5)
        t.bump_fragment("acme", -2)
        assert t.fragment_count == {"acme": 3}

    def test_bump_operation_creates_entry(self):
        t = TenantMetrics()
        t.bump_operation("acme", 1)
        assert t.operation_count == {"acme": 1}

    def test_bump_operation_multiple(self):
        t = TenantMetrics()
        t.bump_operation("acme", 3)
        t.bump_operation("globex", 2)
        assert t.operation_count == {"acme": 3, "globex": 2}


class TestNodeMetricsWithTenant:
    def test_default_tenant_metrics_attached(self):
        registry = MetricsCollector()
        metrics = NodeMetrics(registry=registry)
        assert isinstance(metrics.tenant, TenantMetrics)
        assert metrics.tenant.fragment_count == {}

    def test_node_store_increments_tenant_fragment_count(self):
        from membrane.node import Node

        registry = MetricsCollector()
        node_metrics = NodeMetrics(registry=registry)
        node = Node(node_id="n1", max_memory_bytes=10_000, metrics=node_metrics)
        frag = Fragment(
            identity=_identity(),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        node.store(frag)
        assert node_metrics.tenant.fragment_count == {"acme": 1}

    def test_node_remove_decrements_tenant_fragment_count(self):
        from membrane.node import Node

        registry = MetricsCollector()
        node_metrics = NodeMetrics(registry=registry)
        node = Node(node_id="n1", max_memory_bytes=10_000, metrics=node_metrics)
        frag = Fragment(
            identity=_identity(),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        node.store(frag)
        node.remove_fragment(frag.identity.payload_hash)
        assert node_metrics.tenant.fragment_count == {"acme": 0}

    def test_node_stores_across_tenants(self):
        from membrane.node import Node

        registry = MetricsCollector()
        node_metrics = NodeMetrics(registry=registry)
        node = Node(node_id="n1", max_memory_bytes=100_000, metrics=node_metrics)
        for i, tenant in enumerate(("acme", "globex", "acme", "initrode")):
            frag = Fragment(
                identity=_identity(payload_hash=f"hash{i}" + "0" * 60),
                payload_ref=None,
                payload_size=10,
                ttl=60.0,
                reuse_score=0.5,
                version_id=1,
                tenant_id=tenant,
            )
            node.store(frag)
        assert node_metrics.tenant.fragment_count == {
            "acme": 2,
            "globex": 1,
            "initrode": 1,
        }


class TestClusterMetricsTenantOps:
    def test_cluster_metrics_default(self):
        registry = MetricsCollector()
        cluster_metrics = ClusterMetrics(registry=registry)
        # ClusterMetrics does not own a TenantMetrics in 3.0; the
        # per-tenant operation counter lives in a side-channel
        # dictionary for now (3.5.3 wires it into the proper
        # typed collector). The test asserts the registry does
        # not crash on construction.
        assert isinstance(cluster_metrics.peers_total.value, (int, float))

    def test_sync_tenant_fragment_gauges(self):
        registry = MetricsCollector()
        metrics = NodeMetrics(registry=registry)
        metrics.tenant.bump_fragment("acme", 3)
        metrics.tenant.bump_fragment("globex", 1)
        # The sync method should not raise; it's a no-op for the
        # current Counter primitive.
        metrics.sync_tenant_fragment_gauges()
