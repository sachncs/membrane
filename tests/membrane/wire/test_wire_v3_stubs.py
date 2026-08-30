"""Tests for the wire_v3 generated stubs (Phase 3.3.2)."""

from __future__ import annotations


class TestWireV3Stubs:
    def test_envelope_message_class(self):
        from membrane.wire.v3 import wire_v3_pb2

        env = wire_v3_pb2.Envelope()
        env.request_id = "r1"
        env.tenant_id = "acme"
        env.model_id = "llama-3"
        env.fingerprint_compat = "a" * 64
        env.content_hash = "b" * 64
        env.total_bytes = 100
        env.compression = 1
        env.content_type = ""
        assert env.request_id == "r1"
        assert env.tenant_id == "acme"
        assert env.total_bytes == 100

    def test_tensor_payload(self):
        from membrane.wire.v3 import wire_v3_pb2

        p = wire_v3_pb2.TensorPayload()
        p.data = b"hello"
        p.schema_fingerprint = "c" * 64
        p.shape.extend([1, 2, 3, 4])
        p.dtype = "float16"
        assert bytes(p.data) == b"hello"
        assert list(p.shape) == [1, 2, 3, 4]
        assert p.dtype == "float16"

    def test_chunk_message(self):
        from membrane.wire.v3 import wire_v3_pb2

        c = wire_v3_pb2.Chunk()
        c.chunk_index = 0
        c.offset = 0
        c.data = b"chunk-bytes"
        c.sha256_hex = "d" * 64
        c.is_last = True
        assert c.chunk_index == 0
        assert c.is_last is True

    def test_chunk_request(self):
        from membrane.wire.v3 import wire_v3_pb2

        r = wire_v3_pb2.ChunkRequest()
        r.chunk_index = 5
        r.offset = 1024
        r.length = 4096
        assert r.chunk_index == 5
        assert r.length == 4096

    def test_grpc_stub_class_exists(self):
        from membrane.wire.v3 import wire_v3_pb2_grpc

        assert hasattr(wire_v3_pb2_grpc, "TransferStub")
        assert hasattr(wire_v3_pb2_grpc, "TransferServicer")
        assert hasattr(wire_v3_pb2_grpc, "add_TransferServicer_to_server")
