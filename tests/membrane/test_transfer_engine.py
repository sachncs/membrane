"""Tests for Phase 4 memory pool + transfer engine (Phase 4)."""

from __future__ import annotations

import pytest

from membrane.transfer_engine import (
    CompressionTransport,
    CudaMemoryPool,
    KVTransferEngine,
    RdmaMemoryPool,
    TensorHandle,
    TransferEnvelope,
)


class TestTensorHandle:
    def test_round_trip(self):
        h = TensorHandle(data=b"hello", shape=(2, 3), dtype="float16")
        assert h.tobytes() == b"hello"
        assert h.size_bytes == 5
        assert h.shape == (2, 3)
        assert h.dtype == "float16"


class TestCudaMemoryPool:
    def test_allocates_on_cpu_when_torch_missing(self):
        pool = CudaMemoryPool(device="cpu")
        h = pool.alloc(shape=(4, 8), dtype="float32")
        assert h.shape == (4, 8)
        assert h.dtype == "float32"
        assert len(h.tobytes()) == 4 * 8 * 4

    def test_copy_to_validates_size(self):
        pool = CudaMemoryPool()
        src = pool.alloc((2, 2), "float32")
        dst = pool.alloc((2, 2), "float32")
        pool.copy_to(src, dst)
        pool.copy_to(pool.alloc((2, 2), "float32"), pool.alloc((2, 2), "float32"))
        with pytest.raises(ValueError, match="shape mismatch"):
            pool.copy_to(pool.alloc((2, 2), "float32"), pool.alloc((2, 4), "float32"))

    def test_pin_host_returns_fresh_handle(self):
        pool = CudaMemoryPool()
        src = pool.alloc((2, 2), "float32")
        pinned = pool.pin_host(src)
        assert pinned.tobytes() == src.tobytes()
        assert pinned.shape == src.shape

    def test_close_blocks_alloc(self):
        pool = CudaMemoryPool()
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.alloc((1, 1), "float32")


class TestRdmaMemoryPool:
    def test_delegates_to_cuda_pool(self):
        rdma = RdmaMemoryPool()
        cuda = CudaMemoryPool()
        h_cuda = cuda.alloc((4, 4), "float32")
        h_rdma = rdma.alloc((4, 4), "float32")
        rdma.copy_to(h_cuda, h_rdma)
        assert h_rdma.tobytes() == h_cuda.tobytes()

    def test_rdma_send_stub_returns_size(self):
        rdma = RdmaMemoryPool()
        h = rdma.alloc((2, 2), "float32")
        assert rdma.rdma_send(h, "peer-host:1234") == h.size_bytes


class TestCompressionTransport:
    def test_raw_round_trip(self):
        t = CompressionTransport(method=CompressionTransport.METHOD_RAW)
        out = t.compress(b"hello")
        assert t.decompress(out) == b"hello"

    def test_deflate_round_trip(self):
        t = CompressionTransport(method=CompressionTransport.METHOD_DEFLATE)
        out = t.compress(b"hello" * 100)
        assert t.decompress(out) == b"hello" * 100
        # Compressed is shorter than the original.
        assert len(out) < 600

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="unknown compression"):
            CompressionTransport(method="brotli").compress(b"x")

    def test_truncated_payload_raises(self):
        with pytest.raises(ValueError, match="too short"):
            CompressionTransport(method=CompressionTransport.METHOD_DEFLATE).decompress(
                b"deflate\x00"
            )

    def test_zstd_requires_optional_dependency(self):
        # The zstd path raises a clear error when zstandard is
        # missing; tests run without the optional package so the
        # branch is exercised here.
        t = CompressionTransport(method=CompressionTransport.METHOD_ZSTD)
        with pytest.raises(RuntimeError, match="zstandard"):
            t.compress(b"x")


class TestKVTransferEngine:
    def test_raw_round_trip(self):
        pool = CudaMemoryPool()
        engine = KVTransferEngine(memory_pool=pool)
        k = pool.alloc((2, 2), "float32")
        v = pool.alloc((2, 2), "float32")
        envelope = engine.transfer_kv(k, v)
        assert envelope.compression == "raw"
        k2, v2 = engine.receive_kv(envelope)
        assert k2.tobytes() == k.tobytes()
        assert v2.tobytes() == v.tobytes()

    def test_compressed_round_trip(self):
        pool = CudaMemoryPool()
        transport = CompressionTransport(method=CompressionTransport.METHOD_DEFLATE)
        engine = KVTransferEngine(memory_pool=pool, transport=transport)
        k = pool.alloc((4, 4), "float32")
        v = pool.alloc((4, 4), "float32")
        envelope = engine.transfer_kv(k, v)
        assert envelope.compression == "deflate"
        k2, v2 = engine.receive_kv(envelope)
        assert k2.tobytes() == k.tobytes()
        assert v2.tobytes() == v.tobytes()

    def test_bad_magic_raises(self):
        pool = CudaMemoryPool()
        engine = KVTransferEngine(memory_pool=pool)
        envelope = TransferEnvelope(
                    compressed=bytes.fromhex("01" + (12).to_bytes(4, "little").hex() + b"badmagicdata".hex()),
            compression="raw",
            shape=(2, 2),
            dtype="float32",
        )
        with pytest.raises(ValueError, match="bad magic"):
            engine.receive_kv(envelope)

    def test_quantized_round_trip(self):
        from membrane.quantization import Int8PerChannelQuantizer

        pool = CudaMemoryPool()
        engine = KVTransferEngine(
            memory_pool=pool,
            quantizer=Int8PerChannelQuantizer(),
        )
        k = pool.alloc((4, 4), "float32")
        v = pool.alloc((4, 4), "float32")
        envelope = engine.transfer_kv(k, v)
        k2, v2 = engine.receive_kv(envelope)
        # The round-trip preserves shape across the wire.
        assert k2.shape == (4, 4)
        assert v2.shape == (4, 4)
