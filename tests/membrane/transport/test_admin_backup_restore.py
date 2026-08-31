"""End-to-end /admin/backup + /admin/restore integration test (Phase 3.0+ follow-up).

The /admin/backup and /admin/restore routes ship in the v3.0+
admin surface; this test exercises the full round-trip:
* Store 3 fragments.
* Call /admin/backup; the snapshot file contains 3
  entries.
* Clear the Node.
* Call /admin/restore; the 3 fragments come back.
* Both /admin/backup and /admin/restore record audit
  entries in the chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestAdminBackupRestore:
    def test_full_backup_restore_round_trip(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog, verify_chain
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        # 1. Build a Node + AuditLog and store 3 fragments.
        node = Node(node_id="n1", max_memory_bytes=10_000)
        log = AuditLog()
        for i in range(3):
            ident = PayloadIdentity(
                payload_hash=f"backup-{i}".rjust(64, "0")[:64],
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
        assert node.get_stats().fragment_count == 3

        # 2. Mount admin routes.
        app = FastAPI()
        app.state.node = node
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)

        # 3. Backup.
        backup_path = tmp_path / "snapshot.json"
        resp = client.post("/admin/backup", json={"destination": str(backup_path)})
        assert resp.status_code == 200
        assert resp.json() == {
            "destination": str(backup_path),
            "fragments": 3,
        }
        snapshot = json.loads(backup_path.read_text())
        assert snapshot["node_id"] == "n1"
        assert len(snapshot["fragments"]) == 3

        # 4. Clear the Node + restore from the snapshot.
        for h in list(node.fragments.keys()):
            node.remove_fragment(h)
        assert node.get_stats().fragment_count == 0
        resp = client.post(
            "/admin/restore", json={"source": str(backup_path)}
        )
        assert resp.status_code == 200
        assert resp.json() == {"source": str(backup_path), "restored": 3}
        assert node.get_stats().fragment_count == 3

        # 5. The audit log records the chain.
        assert verify_chain(log.all()) is None
        actions = [e.action for e in log.all()]
        assert "admin.backup" in actions
        assert "admin.restore" in actions

    def test_restore_missing_source_returns_404(self, tmp_path):
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
            "/admin/restore", json={"source": str(tmp_path / "missing.json")}
        )
        assert resp.status_code == 404

    def test_restore_requires_source(self, tmp_path):
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
        resp = client.post("/admin/restore", json={})
        assert resp.status_code == 400
