"""Tests for GrpcServer."""

import threading
import time

import pytest

from membrane.compute.cpu_backend import CPUBackend
from membrane.membrane_node import MembraneNode
from membrane.transport.grpc_server import GrpcServer


class TestGrpcServer:
    """Test suite for gRPC transport."""

    @pytest.fixture(scope="class")
    def server(self):
        node = MembraneNode("grpc-test", max_memory_bytes=10000)
        backend = CPUBackend()
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

        frag = membrane_pb2.FragmentMessage(
            content_hash="grpc-frag2",
            embedding=[0.1, 0.2],
            model_id="m",
            layer_start=0,
            layer_end=1,
            token_start=0,
            token_end=10,
            size=100,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )
        store_resp = stub.StoreFragment(
            membrane_pb2.StoreRequest(fragment=frag, node_id="grpc-test", is_primary=True)
        )
        assert store_resp.success is True

        retrieve_resp = stub.RetrieveFragment(
            membrane_pb2.RetrieveRequest(content_hash="grpc-frag2", node_id="grpc-test")
        )
        assert retrieve_resp.found is True
        assert retrieve_resp.fragment.content_hash == "grpc-frag2"

    def test_prefill_uses_injected_backend(self, server):
        import grpc
        from membrane.transport.proto import membrane_pb2, membrane_pb2_grpc

        channel = grpc.insecure_channel("127.0.0.1:50053")
        stub = membrane_pb2_grpc.MembraneStub(channel)
        resp = stub.Prefill(
            membrane_pb2.PrefillRequest(prompt_tokens=[1, 2, 3], model_id="m", node_id="grpc-test")
        )
        assert resp.success is True
        assert len(resp.fragments) > 0
