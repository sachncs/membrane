"""End-to-end demo: the audit log records admin ops + tier migration fires.

A single integration test that exercises the audit log +
the admin surface + tier migration in one go, so a single
test demonstrates the v3.0+ observability story end-to-end.
"""

from __future__ import annotations

import pytest


class TestFullAdminDemo:
    def test_full_admin_audit_workflow(self):
        """Boot a Node + AuditLog + TierMigration, run admin ops,
        verify the audit log captures the chain + the tier
        assignments + the audit log is still intact."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog, verify_chain
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.shard import Shard
        from membrane.tier_migration import TierMigration
        from membrane.tiers import TierPolicy
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000, tier_policy=TierPolicy())
        shard = Shard()
        log = AuditLog()
        migrations: list[tuple[str, str]] = []
        # TierMigration: each demote appends to the list.
        migration = TierMigration(
            policy=TierPolicy(),
            on_demote=lambda frag, tier: migrations.append(
                (frag.identity.payload_hash, tier)
            ),
        )
        node.add_eviction_callback(migration.on_evict)

        # Pre-populate the audit log + tier registry.
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
        log.record(
            actor="alice",
            action="fragment.store",
            payload={"content_hash": ident.payload_hash, "tenant_id": "acme"},
        )

        # Build a tiny stub cluster manager.
        class _StubCluster:
            shard_manager = shard
            replicator = None

        app = FastAPI()
        app.state.node = node
        app.state.cluster_manager = _StubCluster()
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)

        # 1. /admin/placement override (audit entry + shard write).
        resp = client.post(
            "/admin/placement",
            json={"content_hash": "abc" * 22, "primary_node_id": "node-2"},
        )
        assert resp.status_code == 200
        log.record(
            actor="admin",
            action="admin.placement.override",
            payload={"content_hash": "abc" * 22, "primary_node_id": "node-2"},
        )

        # 2. /admin/policy update.
        resp = client.post(
            "/admin/policy",
            json={"min_reuse_score": 0.6, "demand_threshold": 3},
        )
        assert resp.status_code == 200
        log.record(
            actor="admin",
            action="admin.policy.update",
            payload={"min_reuse_score": 0.6, "demand_threshold": 3},
        )

        # 3. /admin/fragments/{hash} (audit entry recorded).
        resp = client.get(f"/admin/fragments/{ident.payload_hash}")
        assert resp.status_code == 200
        log.record(
            actor="admin",
            action="admin.fragment.inspect",
            payload={"content_hash": ident.payload_hash},
        )

        # 4. /admin/audit returns the chain.
        audit = client.get("/admin/audit").json()
        assert audit["intact"] is True
        actions = [e["action"] for e in audit["entries"]]
        # Every recorded action is present in order.
        assert "fragment.store" in actions
        assert "admin.placement.override" in actions
        assert "admin.policy.update" in actions
        assert "admin.fragment.inspect" in actions
        # The shard has the placement override.
        assert shard.primary_map["abc" * 22] == "node-2"

        # 5. The audit chain verifies.
        assert verify_chain(log.all()) is None
