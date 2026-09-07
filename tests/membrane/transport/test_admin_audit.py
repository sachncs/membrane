"""Tests for the /admin/audit HTTP surface (Phase 3.2.8 follow-up)."""

from __future__ import annotations

import pytest


class TestAdminAuditEndpoint:
    def test_endpoint_returns_intact_flag_with_empty_log(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.transport.admin import create_admin_router

        app = FastAPI()
        app.state.audit_log = AuditLog()
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.get("/admin/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["intact"] is True
        assert body["entries"] == []

    def test_endpoint_reports_intact_after_appends(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.transport.admin import create_admin_router

        log = AuditLog()
        log.record(actor="alice", action="admin.fragment.inspect", payload={"h": "x"})
        log.record(actor="bob", action="admin.evict", payload={"h": "y"})
        app = FastAPI()
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.get("/admin/audit")
        body = resp.json()
        assert body["intact"] is True
        assert len(body["entries"]) == 2
        assert body["entries"][0]["actor"] == "alice"
        assert body["entries"][0]["action"] == "admin.fragment.inspect"
        assert body["entries"][0]["payload"] == {"h": "x"}
        # Each entry carries the prev_hash + entry_hash chain.
        for entry in body["entries"]:
            assert "prev_hash" in entry
            assert "entry_hash" in entry

    def test_endpoint_detects_tampering(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditEntry, AuditLog
        from membrane.transport.admin import create_admin_router

        log = AuditLog()
        log.record(actor="alice", action="admin.fragment.inspect", payload={"h": "x"})
        # Tamper with the recorded entries by replacing one of
        # them with a different actor.
        log._entries[0] = AuditEntry(
            index=0,
            timestamp=log._entries[0].timestamp,
            actor="mallory",
            action="admin.fragment.inspect",
            payload={"h": "x"},
            prev_hash=log._entries[0].prev_hash,
            entry_hash=log._entries[0].entry_hash,
        )
        app = FastAPI()
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.get("/admin/audit")
        body = resp.json()
        assert body["intact"] is False

    def test_endpoint_503_when_audit_log_unconfigured(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.transport.admin import create_admin_router

        app = FastAPI()
        app.include_router(create_admin_router())
        client = TestClient(app)
        # No app.state.audit_log is set; the handler should 503.
        resp = client.get("/admin/audit")
        assert resp.status_code == 503
