"""Tests for HTTPServer."""

import json
import urllib.request

import pytest

from membrane.compute.cpu_backend import CPUBackend
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature
from membrane.transport.http_server import HTTPServer


def make_fragment(content_hash: str = "h1"):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=10,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestHTTPServer:
    """Test suite for HTTPServer."""

    @pytest.fixture(scope="class")
    def server(self):
        node = MembraneNode("test-http", max_memory_bytes=10000)
        srv = HTTPServer(node=node, host="127.0.0.1", port=18080, compute_backend=CPUBackend())
        srv.run_in_thread()
        import time

        time.sleep(0.2)
        yield srv
        srv.stop()

    def test_heartbeat(self, server):
        req = urllib.request.Request("http://127.0.0.1:18080/heartbeat")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            assert data["node_id"] == "test-http"
            assert data["healthy"] is True
            assert "load" in data

    def test_store_and_retrieve(self, server):
        frag = make_fragment("store-test")
        payload = json.dumps(
            {
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
        ).encode()

        req = urllib.request.Request(
            "http://127.0.0.1:18080/store",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            assert data["success"] is True

        req = urllib.request.Request("http://127.0.0.1:18080/retrieve?content_hash=store-test")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            assert data["found"] is True
            assert data["fragment"]["content_hash"] == "store-test"

    def test_retrieve_missing(self, server):
        req = urllib.request.Request("http://127.0.0.1:18080/retrieve?content_hash=missing")
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                assert data["found"] is False
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_metrics(self, server):
        req = urllib.request.Request("http://127.0.0.1:18080/metrics")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            assert "memory_used_bytes" in data
            assert "fragment_count" in data

    def test_inventory(self, server):
        req = urllib.request.Request("http://127.0.0.1:18080/inventory")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            assert "digest" in data

    def test_prefill(self, server):
        payload = json.dumps({"prompt_tokens": list(range(50)), "model_id": "m"}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:18080/prefill",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            assert data["success"] is True
            assert len(data["fragments"]) > 0
