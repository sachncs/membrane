"""End-to-end /admin/placement override test (Phase 3.2.6 follow-up).

The Phase 3.2.6 commit shipped /admin/placement but the
existing test only verified the 503 path (no cluster
manager). This test wires a real Shard manager and asserts
the placement override takes effect end-to-end.
"""

from __future__ import annotations

import pytest


class TestAdminPlacementOverride:
    def test_placement_override_takes_effect(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.shard import Shard
        from membrane.transport.admin import create_admin_router

        shard = Shard()
        app = FastAPI()
        # The admin route reads ``app.state.cluster_manager.shard_manager``
        # so build a tiny stub that exposes the right shape.
        class _StubCluster:
            shard_manager = shard
            replicator = None

        app.state.cluster_manager = _StubCluster()
        app.state.audit_log = AuditLog()
        app.include_router(create_admin_router())
        client = TestClient(app)

        # 1. Confirm the initial primary is unset.
        assert shard.primary_map.get("abc") is None

        # 2. Override primary for "abc" to "node-2".
        resp = client.post(
            "/admin/placement",
            json={"content_hash": "abc" * 22, "primary_node_id": "node-2"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "content_hash": "abc" * 22,
            "primary_node_id": "node-2",
        }
        # The override took effect on the real Shard manager.
        assert shard.primary_map["abc" * 22] == "node-2"

    def test_placement_override_writes_audit_entry(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.shard import Shard
        from membrane.transport.admin import create_admin_router

        class _StubCluster:
            shard_manager = Shard()
            replicator = None

        log = AuditLog()
        app = FastAPI()
        app.state.cluster_manager = _StubCluster()
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.post(
            "/admin/placement",
            json={"content_hash": "h" * 64, "primary_node_id": "node-2"},
        )
        assert resp.status_code == 200
        audit = client.get("/admin/audit").json()
        actions = [e["action"] for e in audit["entries"]]
        assert "admin.placement.override" in actions

    def test_placement_override_chains_audit_with_subsequent_audit_action(self):
        """The hash-chained audit log preserves order across admin ops."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.shard import Shard
        from membrane.transport.admin import create_admin_router

        class _StubCluster:
            shard_manager = Shard()
            replicator = None

        log = AuditLog()
        app = FastAPI()
        app.state.cluster_manager = _StubCluster()
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)
        # 1. First admin op.
        client.post(
            "/admin/placement",
            json={"content_hash": "a" * 64, "primary_node_id": "n1"},
        )
        # 2. Second admin op.
        client.post(
            "/admin/policy",
            json={"min_reuse_score": 0.7, "demand_threshold": 3},
        )
        audit = client.get("/admin/audit").json()
        # Both actions are present in order.
        actions = [e["action"] for e in audit["entries"]]
        assert actions.index("admin.placement.override") < actions.index(
            "admin.policy.update"
        )
        # The chain verifies.
        from membrane.audit import verify_chain

        assert verify_chain(log.all()) is None
