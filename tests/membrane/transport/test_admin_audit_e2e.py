"""End-to-end audit log integration with /admin/audit (Phase 3.2.8 follow-up)."""

from __future__ import annotations

import pytest


class TestAdminAuditEndToEnd:
    def test_admin_evict_writes_audit_entry(self):
        """A successful admin evict fires an audit entry the /admin/audit route returns."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        # 1. Set up Node + AuditLog.
        node = Node(node_id="n1", max_memory_bytes=10_000)
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

        # 2. Build a fake admin auth context that records to the
        # audit log on every operation. The /admin/audit route
        # returns the log.
        app = FastAPI()
        app.state.node = node
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)

        # 3. Evict the fragment via /admin/evict. The current
        # admin router writes the audit entry as 'admin.evict'
        # attributed to the (empty) caller subject; the
        # test asserts the entry is present.
        resp = client.post(
            "/admin/evict", json={"content_hash": ident.payload_hash}
        )
        assert resp.status_code == 200

        # 4. Query /admin/audit and verify the entry was recorded.
        audit_resp = client.get("/admin/audit")
        assert audit_resp.status_code == 200
        body = audit_resp.json()
        assert body["intact"] is True
        actions = [e["action"] for e in body["entries"]]
        assert "admin.evict" in actions

    def test_admin_inspect_writes_audit_entry(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
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
        app = FastAPI()
        app.state.node = node
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)

        resp = client.get(f"/admin/fragments/{ident.payload_hash}")
        assert resp.status_code == 200

        audit_resp = client.get("/admin/audit")
        actions = [e["action"] for e in audit_resp.json()["entries"]]
        assert "admin.fragment.inspect" in actions

    def test_admin_policy_update_writes_audit_entry(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        log = AuditLog()
        app = FastAPI()
        app.state.node = node
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)

        resp = client.post(
            "/admin/policy",
            json={"min_reuse_score": 0.5, "demand_threshold": 10},
        )
        assert resp.status_code == 200

        audit_resp = client.get("/admin/audit")
        actions = [e["action"] for e in audit_resp.json()["entries"]]
        assert "admin.policy.update" in actions
        # The payload carries the new values.
        for entry in audit_resp.json()["entries"]:
            if entry["action"] == "admin.policy.update":
                assert entry["payload"]["min_reuse_score"] == 0.5
                assert entry["payload"]["demand_threshold"] == 10
                break
