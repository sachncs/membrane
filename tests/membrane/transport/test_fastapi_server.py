"""Tests for FastAPIServer."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from membrane.compute.cpu_backend import CPUBackend
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature
from membrane.transfer_service import TransferService
from membrane.transport.fastapi_server import FastAPIServer, create_app


def make_fragment(content_hash: str = "h1"):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 10)
        ),
        size=100,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestFastAPIServer:
    """Test suite for FastAPI transport endpoints."""

    @pytest.fixture
    def client(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        backend = CPUBackend()
        transfer = TransferService()
        app = create_app(node=node, compute_backend=backend, transfer_service=transfer, cluster_manager=None)
        return TestClient(app)

    def test_heartbeat(self, client):
        resp = client.get("/heartbeat")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "n1"
        assert data["healthy"] is True

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "n1"
        assert "fragment_count" in data

    def test_store_and_retrieve(self, client):
        frag = make_fragment("abc")
        payload = {
            "fragment": {
                "content_hash": frag.content_hash,
                "embedding": list(frag.embedding),
                "model_id": frag.structural_signature.model_id,
                "layer_range": list(frag.structural_signature.layer_range),
                "token_span": list(frag.structural_signature.token_span),
                "size": frag.size,
                "ttl": frag.ttl,
                "reuse_score": frag.reuse_score,
                "version_id": frag.version_id,
            },
            "is_primary": True,
        }
        resp = client.post("/store", json=payload)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        resp = client.get("/retrieve?content_hash=abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["fragment"]["content_hash"] == "abc"

    def test_retrieve_not_found(self, client):
        resp = client.get("/retrieve?content_hash=missing")
        assert resp.status_code == 200
        assert resp.json()["found"] is False

    def test_inventory(self, client):
        frag = make_fragment("inv1")
        client.post("/store", json={
            "fragment": {
                "content_hash": frag.content_hash,
                "embedding": list(frag.embedding),
                "model_id": frag.structural_signature.model_id,
                "layer_range": list(frag.structural_signature.layer_range),
                "token_span": list(frag.structural_signature.token_span),
                "size": frag.size,
                "ttl": frag.ttl,
                "reuse_score": frag.reuse_score,
                "version_id": frag.version_id,
            },
            "is_primary": True,
        })
        resp = client.get("/inventory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "n1"
        assert "inv1" in data["digest"]

    def test_prefill(self, client):
        resp = client.post("/prefill", json={"prompt_tokens": [1, 2, 3], "model_id": "m"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["fragments"]) > 0

    def test_replicate(self, client):
        frag = make_fragment("rep1")
        resp = client.post("/replicate", json={
            "fragment": {
                "content_hash": frag.content_hash,
                "embedding": list(frag.embedding),
                "model_id": frag.structural_signature.model_id,
                "layer_range": list(frag.structural_signature.layer_range),
                "token_span": list(frag.structural_signature.token_span),
                "size": frag.size,
                "ttl": frag.ttl,
                "reuse_score": frag.reuse_score,
                "version_id": frag.version_id,
            },
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_join_leave_without_cluster_manager(self, client):
        resp = client.post("/join", json={"node_id": "n2", "host": "127.0.0.1", "port": 8081})
        assert resp.status_code == 200
        assert resp.json()["error"] == "cluster manager not enabled"

        resp = client.post("/leave", json={"node_id": "n2"})
        assert resp.status_code == 200
        assert resp.json()["error"] == "cluster manager not enabled"

    def test_gossip_without_cluster_manager(self, client):
        resp = client.post("/gossip", json={"node_id": "n2", "timestamp": 1.0, "peers": []})
        assert resp.status_code == 200
        assert resp.json()["error"] == "cluster manager not enabled"

    def test_peers_without_cluster_manager(self, client):
        resp = client.get("/peers")
        assert resp.status_code == 200
        assert resp.json()["error"] == "cluster manager not enabled"

    def test_join_with_cluster_manager(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        backend = CPUBackend()
        transfer = TransferService()
        cluster = MagicMock()
        cluster.on_peer_join.return_value = {"success": True, "peers": []}
        app = create_app(node=node, compute_backend=backend, transfer_service=transfer, cluster_manager=cluster)
        client = TestClient(app)
        resp = client.post("/join", json={"node_id": "n2", "host": "127.0.0.1", "port": 8081})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        cluster.on_peer_join.assert_called_once_with("n2", "127.0.0.1", 8081)

    def test_server_start_stop(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        srv = FastAPIServer(node=node, host="127.0.0.1", port=18080)
        import threading
        t = threading.Thread(target=srv.start, daemon=True)
        t.start()
        import time
        time.sleep(0.5)
        srv.stop()
        t.join(timeout=2)
        assert not t.is_alive()
