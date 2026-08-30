"""Tests for GrpcServer."""

import threading
import time

import pytest

from membrane.compute.cpu import CPU
from membrane.node import Node
from membrane.transport.grpc import GrpcServer


def _identity_payload(payload_hash_hex: str, model_id: str = "m") -> dict:
    """Build a wire-friendly identity dict for gRPC test fixtures."""
    return {
        "payload_hash": payload_hash_hex,
        "model_id": model_id,
        "model_revision": "",
        "tokenizer_name": model_id,
        "tokenizer_revision": "",
        "layer_range": [0, 1],
        "head_range": [-1, -1],
        "token_span": [0, 10],
        "dtype": "float16",
        "shape": [1, 1, 11, 1, 64],
    }


_HASH_HEX = "ab" * 32  # 64 hex chars = 32 raw bytes, looks like a sha256 digest


class TestGrpcServer:
    """Test suite for gRPC transport."""

    @pytest.fixture(scope="class")
    @classmethod
    def server(cls):
        node = Node("grpc-test", max_memory_bytes=10000)
        backend = CPU()
        srv = GrpcServer(node=node, host="127.0.0.1", port=50053, compute_backend=backend)
        t = threading.Thread(target=srv.start, daemon=True)
        t.start()
        time.sleep(0.5)
        yield srv
        srv.stop()

    def test_heartbeat(self, server):
        import grpc

        from membrane.transport.proto import membrane_pb2, membrane_pb2_grpc

        channel = grpc.insecure_channel("127.0.0.1:50053")
        stub = membrane_pb2_grpc.MembraneStub(channel)
        resp = stub.Heartbeat(membrane_pb2.HeartbeatRequest(node_id="grpc-test"))
        assert resp.healthy is True
        assert resp.node_id == "grpc-test"

    def test_store_and_retrieve(self, server):
        import grpc

        from membrane.transport.proto import membrane_pb2, membrane_pb2_grpc

        channel = grpc.insecure_channel("127.0.0.1:50053")
        stub = membrane_pb2_grpc.MembraneStub(channel)

        ident = _identity_payload(_HASH_HEX)
        frag = membrane_pb2.FragmentMessage(
            schema_version=2,
            payload_hash=bytes.fromhex(_HASH_HEX),
            model_id=ident["model_id"],
            model_revision=ident["model_revision"],
            tokenizer_name=ident["tokenizer_name"],
            tokenizer_revision=ident["tokenizer_revision"],
            layer_range=ident["layer_range"],
            head_range=ident["head_range"],
            token_span=ident["token_span"],
            dtype=ident["dtype"],
            shape=ident["shape"],
            payload_ref=_HASH_HEX,
            payload_size=100,
            payload=b"",
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )
        store_resp = stub.StoreFragment(membrane_pb2.StoreRequest(fragment=frag, node_id="grpc-test", is_primary=True))
        assert store_resp.success is True

        retrieve_resp = stub.RetrieveFragment(
            membrane_pb2.RetrieveRequest(content_hash=_HASH_HEX, node_id="grpc-test")
        )
        assert retrieve_resp.found is True
        assert retrieve_resp.fragment.payload_hash == bytes.fromhex(_HASH_HEX)

    def test_prefill_uses_injected_backend(self, server):
        import grpc

        from membrane.transport.proto import membrane_pb2, membrane_pb2_grpc

        channel = grpc.insecure_channel("127.0.0.1:50053")
        stub = membrane_pb2_grpc.MembraneStub(channel)
        resp = stub.Prefill(membrane_pb2.PrefillRequest(prompt_tokens=[1, 2, 3], model_id="m", node_id="grpc-test"))
        assert resp.success is True
        assert len(resp.fragments) > 0
