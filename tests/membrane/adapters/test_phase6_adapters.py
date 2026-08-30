"""Tests for the SGLang and TensorRT-LLM adapters (Phase 6)."""

from __future__ import annotations

import pytest

from membrane.adapters.sglang import (
    SGLANG_AVAILABLE,
    InMemorySGLangClient,
    MembraneSGLangAdapter,
    SGLangKVEntry,
)
from membrane.adapters.trtllm import (
    TRTLLM_AVAILABLE,
    InMemoryTrtClient,
    MembraneTrtAdapter,
    TrtKVBlock,
)


class _FakeSGLangPool:
    """Stub pool that supports the ``k_buffer`` / ``v_buffer`` path."""

    def __init__(self) -> None:
        self.k_buffer: dict[int, bytes] = {0: b"k0", 1: b"k1"}
        self.v_buffer: dict[int, bytes] = {0: b"v0", 1: b"v1"}


class _FakeSGLangModel:
    def __init__(self) -> None:
        self.kv_pool = _FakeSGLangPool()


class TestMembraneSGLangAdapter:
    def test_extract_returns_empty_when_no_pool(self):
        adapter = MembraneSGLangAdapter(kv_backend=None)

        class _NoPool:
            pass

        bundle = adapter.extract(_NoPool(), (0, 0), (-1, -1), (0, 0))
        assert bundle.layers == ()

    def test_extract_and_round_trip(self):
        adapter = MembraneSGLangAdapter(kv_backend=None)
        bundle = adapter.extract(_FakeSGLangModel(), (0, 0), (-1, -1), (0, 0))
        assert len(bundle.layers) == 1
        assert bytes(bundle.layers[0].k) == b"k0"
        assert bytes(bundle.layers[0].v) == b"v0"

    def test_import_into_no_pool_is_noop(self):
        adapter = MembraneSGLangAdapter(kv_backend=None)
        from membrane.adapters import KVTensor

        class _NoPool:
            pass

        empty = KVTensor(
            layers=(),
            layer_range=(0, 0),
            head_range=(-1, -1),
            token_span=(0, 0),
            shape=(1, 1, 1, 64),
            fingerprint=None,
        )
        adapter.import_into(_NoPool(), empty, (0, 0))


class TestInMemorySGLangClient:
    def test_seed_and_get(self):
        client = InMemorySGLangClient()
        entries = (SGLangKVEntry(token_id=1, k=b"k1", v=b"v1"),)
        client.seed("m", "h", entries)
        assert client.get("m", "h") == entries
        assert client.get("m", "missing") == ()


@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="sglang not installed")
def test_sglang_real_pool_class_imported():
    from sglang.srt.mem_cache.memory_pool import TokenToKVPool

    assert TokenToKVPool is not None


class _FakeTrtManager:
    def __init__(self) -> None:
        self.k_cache: dict[int, dict[int, bytes]] = {0: {0: b"k0", 1: b"k1"}}
        self.v_cache: dict[int, dict[int, bytes]] = {0: {0: b"v0", 1: b"v1"}}


class _FakeTrtModel:
    def __init__(self) -> None:
        self.kv_cache_manager = _FakeTrtManager()


class TestMembraneTrtAdapter:
    def test_blocks_for_token_span(self):
        adapter = MembraneTrtAdapter(kv_backend=None)
        assert adapter._blocks_for((0, 63)) == (0,)
        assert adapter._blocks_for((0, 64)) == (0, 1)
        assert adapter._blocks_for((64, 127)) == (1,)

    def test_extract_returns_empty_when_no_manager(self):
        adapter = MembraneTrtAdapter(kv_backend=None)

        class _NoManager:
            pass

        bundle = adapter.extract(_NoManager(), (0, 0), (-1, -1), (0, 0))
        assert bundle.layers == ()

    def test_extract_and_round_trip(self):
        adapter = MembraneTrtAdapter(kv_backend=None)
        bundle = adapter.extract(_FakeTrtModel(), (0, 0), (-1, -1), (0, 63))
        assert len(bundle.layers) == 1
        assert bytes(bundle.layers[0].k) == b"k0"
        assert bytes(bundle.layers[0].v) == b"v0"

    def test_import_into_no_manager_is_noop(self):
        adapter = MembraneTrtAdapter(kv_backend=None)
        from membrane.adapters import KVTensor

        class _NoManager:
            pass

        empty = KVTensor(
            layers=(),
            layer_range=(0, 0),
            head_range=(-1, -1),
            token_span=(0, 0),
            shape=(1, 1, 1, 64),
            fingerprint=None,
        )
        adapter.import_into(_NoManager(), empty, (0, 0))


class TestInMemoryTrtClient:
    def test_seed_and_get(self):
        client = InMemoryTrtClient()
        blocks = (TrtKVBlock(block_id=0, k=b"k0", v=b"v0"),)
        client.seed("m", "h", blocks)
        assert client.get("m", "h") == blocks
        assert client.get("m", "missing") == ()


@pytest.mark.skipif(not TRTLLM_AVAILABLE, reason="tensorrt_llm not installed")
def test_trtllm_real_kv_cache_manager_imported():
    from tensorrt_llm.runtime.kv_cache_manager import KVCacheManager

    assert KVCacheManager is not None
