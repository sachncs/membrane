"""Tests for the vLLM KVConnector adapter (Phase 5)."""

from __future__ import annotations

from typing import Any

import pytest

from membrane.adapters.vllm import (
    VLLM_AVAILABLE,
    InMemoryClusterClient,
    MembraneVLLMAdapter,
    MembraneVLLMConnector,
)


class _Request:
    """Minimal vLLM ``Request`` shim for tests."""

    def __init__(self, request_id: str, token_ids: tuple[int, ...], model_id: str = "m") -> None:
        self.request_id = request_id
        self.token_ids = list(token_ids)
        self.model_id = model_id


class _PageAttnParams:
    """Minimal vLLM ``PageAttentionParams`` shim for tests."""

    pass


def _build_connector() -> Any:
    adapter = MembraneVLLMAdapter(kv_backend=None)
    client = InMemoryClusterClient()
    return adapter.make_connector(client=client, n_layers=4, model_id="llama-7b", dtype="float16")


class TestInMemoryClusterClient:
    def test_lookup_miss_returns_zero(self):
        client = InMemoryClusterClient()
        result = client.lookup_prefix("m", (1, 2, 3))
        assert result.prefix_len == 0
        assert result.kv_handle == ""

    def test_lookup_hit_after_seed(self):
        client = InMemoryClusterClient()
        client.seed("mem:m:3", {0: b"k0v0", 1: b"k1v1"})
        result = client.lookup_prefix("m", (1, 2, 3))
        assert result.prefix_len == 3
        assert result.kv_handle == "mem:m:3"

    def test_start_load_filters_missing_layers(self):
        client = InMemoryClusterClient()
        client.seed("mem:m:3", {0: b"k0v0", 2: b"k2v2"})
        loads = client.start_load("mem:m:3", (0, 1, 2, 3))
        assert [load.layer_idx for load in loads] == [0, 2]

    def test_fetch_layer_returns_single_layer_bundle(self):
        client = InMemoryClusterClient()
        client.seed("mem:m:3", {1: b"k1v1"})
        load = LayerLoadStub(layer_idx=1, kv_handle="mem:m:3")  # type: ignore[arg-type]
        bundle = client.fetch_layer(load, "m", (1, 1, 1, 64), "float16")
        assert len(bundle.layers) == 1
        assert bundle.layers[0].layer_idx == 1
        assert bytes(bundle.layers[0].k) == b"k1v1"

    def test_save_layer_round_trips(self):
        client = InMemoryClusterClient()
        from membrane.adapters import LayerKV

        client.save_layer(
            LayerKV(layer_idx=0, k=b"k0", v=b"v0", head_range=(-1, -1), dtype="float16"),
            model_id="m",
            token_span=(0, 1),
        )
        result = client.lookup_prefix("m", (1,))
        assert result.prefix_len == 0
        loads = client.start_load("mem:0", (0,))
        assert [load.layer_idx for load in loads] == [0]


class LayerLoadStub:
    """Plain object that mimics :class:`LayerLoad` for tests."""

    def __init__(self, layer_idx: int, kv_handle: str) -> None:
        self.layer_idx = layer_idx
        self.kv_handle = kv_handle


class TestMembraneVLLMConnector:
    def test_full_load_save_cycle(self):
        connector = _build_connector()
        client = connector.client
        client.seed("mem:llama-7b:4", {0: b"k0", 1: b"k1", 2: b"k2", 3: b"k3"})
        request = _Request("r1", (10, 11, 12, 13), model_id="llama-7b")
        params = _PageAttnParams()
        matched = connector.get_num_new_matched_tokens(request, params)
        assert matched == 4
        connector.update_state_after_alloc(request, (100, 101, 102), params)
        connector.start_load_kv(request)
        for layer in range(4):
            connector.wait_for_layer_load(layer, request)
        meta = connector.build_connector_meta(None)
        assert meta["r1"]["matched_prefix"] == 4
        assert meta["r1"]["block_table"] == [100, 101, 102]

    def test_miss_does_not_start_load(self):
        connector = _build_connector()
        request = _Request("r1", (10, 11, 12, 13), model_id="llama-7b")
        params = _PageAttnParams()
        matched = connector.get_num_new_matched_tokens(request, params)
        assert matched == 0
        connector.start_load_kv(request)
        for layer in range(4):
            connector.wait_for_layer_load(layer, request)

    def test_save_kv_uses_split_path(self):
        connector = _build_connector()
        captured: list[int] = []

        class _CapturingClient(InMemoryClusterClient):
            def save_layer(self, layer: Any, model_id: str, token_span: tuple[int, int]) -> None:  # type: ignore[override]
                captured.append(layer.layer_idx)
                super().save_layer(layer, model_id, token_span)

        connector.client = _CapturingClient()
        kv_cache = _SplitFakeTensor(b"k0k0v0v0")
        connector.save_kv(0, [kv_cache], None)
        assert captured == [0, 0]
        assert 0 in connector.client._by_handle.get("mem:0", {})

    def test_make_connector_returns_vllm_subclass(self):
        if VLLM_AVAILABLE:
            connector = _build_connector()
            from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase
            assert isinstance(connector, KVConnectorBase)
        else:
            assert MembraneVLLMConnector is not None


class _SplitFakeTensor:
    """Tensor-like object with a ``split`` method that returns two halves."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def split(self, split_size: int, dim: int) -> tuple[_SplitFakeTensor, _SplitFakeTensor]:
        half = len(self.payload) // 2
        return _SplitFakeTensor(self.payload[:half]), _SplitFakeTensor(self.payload[half:])

    def contiguous(self) -> _SplitFakeTensor:
        return self

    def numpy(self) -> _SplitFakeTensor:
        return self

    def detach(self) -> _SplitFakeTensor:
        return self

    def cpu(self) -> _SplitFakeTensor:
        return self

    def tobytes(self) -> bytes:
        return self.payload

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _SplitFakeTensor):
            return self.payload == other.payload
        return NotImplemented


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vllm not installed")
def test_vllm_real_base_class():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase

    connector = _build_connector()
    assert isinstance(connector, KVConnectorBase)
