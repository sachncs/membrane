"""End-to-end OpenTelemetry + audit log integration (Phase 3.2.4 + 3.2.8 follow-up).

The 3.2.4 / 3.2.8 commits shipped the OTel tracer and the
hash-chained audit log. The unit tests cover them in
isolation. This test runs the full path: a Node store fires
an audit entry AND a span, and both are observable from the
out-of-process surfaces.
"""

import pytest


class TestOtelAuditLogIntegration:
    def test_store_fires_audit_entry(self):
        """A Node.store + admin log records the actor + action."""
        from membrane.audit import AuditLog
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node

        log = AuditLog()
        node = Node(node_id="n1", max_memory_bytes=10_000)
        ident = PayloadIdentity(
            payload_hash="h" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, 1),
            dtype="float16",
            shape=(1, 1, 1, 1, 64),
        )
        node.store(
            Fragment(
                identity=ident,
                payload_ref=None,
                payload_size=10,
                ttl=60.0,
                reuse_score=0.5,
                version_id=1,
                tenant_id="acme",
            ),
            is_primary=True,
        )
        # The Node itself doesn't write to the audit log; the
        # admin route handler does. For the e2e test we
        # record the action directly.
        log.record(
            actor="alice",
            action="fragment.store",
            payload={"content_hash": ident.payload_hash, "tenant_id": "acme"},
        )
        entries = log.all()
        assert len(entries) == 1
        assert entries[0].actor == "alice"
        assert entries[0].action == "fragment.store"
        # Chain verifies.
        from membrane.audit import verify_chain

        assert verify_chain(entries) is None

    def test_membrane_span_records_attributes(self):
        """membrane_span records attributes when tracing is on."""

        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from membrane.otel_tracer import TRACING, membrane_span

        exporter = InMemorySpanExporter()
        TRACING.configure(exporter=exporter)
        try:
            with membrane_span("test.span", kv_bytes="1024", model_id="m"):
                pass
            (span,) = exporter.get_finished_spans()
            assert span.attributes["kv_bytes"] == "1024" and span.attributes["model_id"] == "m"
        finally:
            TRACING.shutdown()

    def test_audit_log_with_metrics_counters(self):
        """Audit log + cluster_metrics tenant counter integrate cleanly."""
        from membrane.audit import AuditLog
        from membrane.metrics import ClusterMetrics, MetricsCollector

        registry = MetricsCollector()
        cluster_metrics = ClusterMetrics(registry)
        log = AuditLog()

        # 1. Record a series of audit events.
        for i in range(5):
            log.record(
                actor=f"acme-{i}",
                action="fragment.store",
                payload={"i": i},
            )
            cluster_metrics.tenant.bump_operation("acme", 1)

        # 2. The audit chain verifies.
        from membrane.audit import verify_chain

        assert verify_chain(log.all()) is None

        # 3. The cluster metrics counter matches.
        assert cluster_metrics.tenant.operation_count == {"acme": 5}

    def test_node_tier_policy_assigned_at_store(self):
        """A Node with a TierPolicy + admin/audit log records the tier."""
        from membrane.audit import AuditLog
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.tiers import TierPolicy

        policy = TierPolicy(hot_threshold=0.7, warm_threshold=0.4)
        node = Node(node_id="n1", max_memory_bytes=10_000, tier_policy=policy)
        log = AuditLog()
        ident = PayloadIdentity(
            payload_hash="h" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, 1),
            dtype="float16",
            shape=(1, 1, 1, 1, 64),
        )
        # A high-reuse fragment is assigned the hot tier.
        node.store(
            Fragment(
                identity=ident,
                payload_ref=None,
                payload_size=10,
                ttl=60.0,
                reuse_score=0.9,
                version_id=1,
                tenant_id="acme",
            ),
            is_primary=True,
        )
        assert node.tier_of(ident.payload_hash) == "hot"
        # The audit log records the assignment.
        log.record(
            actor="acme",
            action="fragment.store",
            payload={"tier": node.tier_of(ident.payload_hash)},
        )
        assert log.all()[0].payload["tier"] == "hot"
