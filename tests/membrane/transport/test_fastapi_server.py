"""Tests for FastAPIServer."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from membrane.compute.cpu import CPU
from membrane.fragment import Fragment
from membrane.node import Node
from membrane.serialization import to_dict
from membrane.transfer import TransferService
from membrane.transport.fastapi import FastAPIServer, create_app
from tests.conftest import make_fragment


class TestFastAPIServer:
    """Test suite for FastAPI transport endpoints."""

    @pytest.fixture
    def client(self):
        node = Node("n1", max_memory_bytes=10000)
        backend = CPU()
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
        client.app.state.node.content_store.put(frag.payload_ref, b"kv-bytes")
        payload = {
            "fragment": to_dict(frag),
            "is_primary": True,
        }
        resp = client.post("/store", json=payload)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        resp = client.get("/retrieve?content_hash=abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["fragment"]["identity"]["payload_hash"] == "abc"

    def test_store_rejects_missing_payload_bytes(self, client):
        """A fragment whose payload_ref is not in the content store is a 422, not a phantom write."""
        frag = make_fragment("nobytes")
        resp = client.post("/store", json={"fragment": to_dict(frag), "is_primary": True})
        assert resp.status_code == 422
        assert resp.json()["payload_ref"] == frag.payload_ref
        assert client.get("/retrieve?content_hash=nobytes").json()["found"] is False

    def test_retrieve_not_found(self, client):
        resp = client.get("/retrieve?content_hash=missing")
        assert resp.status_code == 200
        assert resp.json()["found"] is False

    def test_inventory(self, client):
        frag = make_fragment("inv1")
        client.app.state.node.content_store.put(frag.payload_ref, b"kv-bytes")
        client.post(
            "/store",
            json={
                "fragment": to_dict(frag),
                "is_primary": True,
            },
        )
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
        import hashlib

        frag = make_fragment("rep1")
        # Metadata without its bytes is refused: the replica could not serve it.
        resp = client.post("/replicate", json={"fragment": to_dict(frag)})
        assert resp.status_code == 422
        assert resp.json()["error"] == "payload missing"

        data = b"kv-bytes" * 8
        bad = client.put(f"/blobs/{frag.payload_ref}", content=data, headers={"X-Content-SHA256": "0" * 64})
        assert bad.status_code == 400
        digest = hashlib.sha256(data).hexdigest()
        put = client.put(f"/blobs/{frag.payload_ref}", content=data, headers={"X-Content-SHA256": digest})
        assert put.status_code == 200 and put.json()["stored"] is True

        resp = client.post("/replicate", json={"fragment": to_dict(frag)})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        got = client.get(f"/blobs/{frag.payload_ref}")
        assert got.content == data and got.headers["x-content-sha256"] == digest
        head = client.head(f"/blobs/{frag.payload_ref}")
        assert head.status_code == 200 and head.headers["x-content-sha256"] == digest
        assert client.get("/blobs/absent").status_code == 404
        assert client.get("/blobs/..").status_code in (400, 404)

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
        node = Node("n1", max_memory_bytes=10000)
        backend = CPU()
        transfer = TransferService()
        cluster = MagicMock()
        cluster.membership.add.return_value = None
        cluster.membership.to_json.return_value = []
        app = create_app(node=node, compute_backend=backend, transfer_service=transfer, cluster_manager=cluster)
        client = TestClient(app)
        resp = client.post("/join", json={"node_id": "n2", "host": "127.0.0.1", "port": 8081})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        cluster.membership.add.assert_called_once_with("n2", "127.0.0.1", 8081, peer_cn="")

    def test_server_start_stop(self):
        node = Node("n1", max_memory_bytes=10000)
        srv = FastAPIServer(node=node, host="127.0.0.1", port=18080)
        import threading

        t = threading.Thread(target=srv.start, daemon=True)
        t.start()
        import time

        time.sleep(0.5)
        srv.stop()
        t.join(timeout=2)
        assert not t.is_alive()


def test_prefill_then_retrieve_round_trips():
    """The first-run path: fragments created by /prefill are retrievable."""
    node = Node("n1", max_memory_bytes=1_000_000)
    app = create_app(node=node, compute_backend=CPU(), transfer_service=TransferService(), cluster_manager=None)
    client = TestClient(app)
    frags = client.post("/prefill", json={"prompt_tokens": list(range(300)), "model_id": "m"}).json()["fragments"]
    assert len(frags) == 3
    for frag in frags:
        content_hash = frag["identity"]["payload_hash"]
        body = client.get(f"/retrieve?content_hash={content_hash}").json()
        assert body["found"] is True
        assert len(node.content_store.get(frag["payload_ref"])) == frag["payload_size"]
