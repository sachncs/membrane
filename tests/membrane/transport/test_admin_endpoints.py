"""Tests for the /admin/placement + /admin/evict + /admin/repair + /admin/policy endpoints."""

from __future__ import annotations

import pytest


class TestAdminEndpoints:
    def test_inspect_returns_fragment_metadata(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

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
                version_id=7,
                tenant_id="acme",
            ),
            is_primary=True,
        )

        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)

        resp = client.get(f"/admin/fragments/{ident.payload_hash}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["content_hash"] == ident.payload_hash
        assert body["tenant_id"] == "acme"
        assert body["version_id"] == 7
        assert body["primary"] is True

    def test_inspect_404(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.get("/admin/fragments/ffffffff")
        assert resp.status_code == 404

    def test_evict_removes_fragment(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

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

        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.post(
            "/admin/evict", json={"content_hash": ident.payload_hash}
        )
        assert resp.status_code == 200
        assert resp.json() == {"content_hash": ident.payload_hash, "evicted": True}
        # The fragment is gone.
        resp2 = client.get(f"/admin/fragments/{ident.payload_hash}")
        assert resp2.status_code == 404

    def test_evict_404(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.post("/admin/evict", json={"content_hash": "x" * 64})
        assert resp.status_code == 404

    def test_placement_override(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = FastAPI()
        app.state.node = node
        # The placement route needs a cluster_manager; when
        # missing the handler returns 503.
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.post(
            "/admin/placement",
            json={"content_hash": "h" * 64, "primary_node_id": "node-2"},
        )
        assert resp.status_code == 503

    def test_repair_returns_503_without_replicator(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.post("/admin/repair", json={"peer_node_id": "peer-1"})
        assert resp.status_code == 503

    def test_policy_get_returns_default(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.get("/admin/policy")
        assert resp.status_code == 200
        body = resp.json()
        assert "min_reuse_score" in body
        assert "demand_threshold" in body

    def test_policy_post_echoes(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.node import Node
        from membrane.transport.admin import create_admin_router

        node = Node(node_id="n1", max_memory_bytes=10_000)
        app = FastAPI()
        app.state.node = node
        app.include_router(create_admin_router())
        client = TestClient(app)
        resp = client.post(
            "/admin/policy",
            json={"min_reuse_score": 0.6, "demand_threshold": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["min_reuse_score"] == 0.6
        assert body["demand_threshold"] == 5
