"""Tests for KVBackend."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from membrane.content_store import InProcessBytes
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity

torch = pytest.importorskip("torch")


class _LayerCache:
    """Per-layer DynamicCache stand-in holding (K, V) tensors."""

    def __init__(self, k: Any, v: Any) -> None:
        self.k = k
        self.v = v


class _DynamicCache:
    """Iterate-from-objects DynamicCache stand-in (transformers v5 API)."""

    def __init__(self, layers: list[_LayerCache]) -> None:
        self.layers = layers

    def __iter__(self) -> Any:
        return iter((layer.k, layer.v) for layer in self.layers)

    def __len__(self) -> int:
        return len(self.layers)


def _make_pkv(batch: int, n_heads: int, seq_len: int, head_dim: int, n_layers: int) -> _DynamicCache:
    layers = []
    for _ in range(n_layers):
        k = torch.zeros((batch, n_heads, seq_len, head_dim), dtype=torch.float16)
        v = torch.ones((batch, n_heads, seq_len, head_dim), dtype=torch.float16)
        layers.append(_LayerCache(k, v))
    return _DynamicCache(layers)


class TestKVBackend:
    """Test suite for KVBackend."""

    def test_unloaded_falls_back_to_simulation(self):
        """When the model is not loaded, prefill returns simulated fragments."""
        from membrane.compute.kv import KVBackend

        store = InProcessBytes()
        backend = KVBackend(content_store=store, model_id="missing-model")
        backend.model = None
        backend.tokenizer = None
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert frags[0].identity.model_id == "m"
        assert frags[0].payload_ref is not None

    def test_simulated_payload_not_in_content_store(self):
        """Simulated fragments don't write to the content store."""
        from membrane.compute.kv import KVBackend

        store = InProcessBytes()
        backend = KVBackend(content_store=store, model_id="missing-model")
        backend.model = None
        backend.prefill([1, 2, 3, 4], "m")
        assert store.size() == 0

    def test_real_kv_round_trip(self):
        """Real K/V tensors land in the content store as canonical frames."""
        from membrane.compute.kv import KVBackend

        store = InProcessBytes()
        backend = KVBackend.__new__(KVBackend)
        backend.content_store = store
        backend.model_id = "fake-model"
        backend.model_revision = "abc123"
        backend.tokenizer_name = "fake-model"
        backend.tokenizer_revision = "abc123"
        backend.dtype = "float16"
        backend.window_size = 128
        backend.n_layers = 2
        backend.n_heads = 4
        backend.head_dim = 8
        backend.model = object()
        backend.tokenizer = object()
        backend.torch = torch

        pkv = _make_pkv(batch=1, n_heads=4, seq_len=10, head_dim=8, n_layers=2)
        seq_len = 10
        frames = backend._frames_for_windows(pkv=pkv, seq_len=seq_len, model_id="fake-model")
        assert len(frames) == 1

        identity, frame = frames[0]
        assert isinstance(identity, PayloadIdentity)
        assert identity.payload_hash == hashlib.sha256(_payload_only(frame, n_layers=2, n_heads=4, window_len=10, head_dim=8)).hexdigest()
        assert identity.token_span == (0, 9)
        assert identity.layer_range == (0, 1)
        assert identity.shape == (1, 2, 4, 10, 8)
        assert identity.dtype == "float16"

    def test_windowing_chunks_long_sequences(self):
        """A prompt longer than window_size produces multiple frames."""
        from membrane.compute.kv import KVBackend

        store = InProcessBytes()
        backend = KVBackend.__new__(KVBackend)
        backend.content_store = store
        backend.model_id = "fake-model"
        backend.model_revision = ""
        backend.tokenizer_name = "fake-model"
        backend.tokenizer_revision = ""
        backend.dtype = "float16"
        backend.window_size = 8
        backend.n_layers = 1
        backend.n_heads = 1
        backend.head_dim = 4
        backend.model = object()
        backend.tokenizer = object()
        backend.torch = torch

        pkv = _make_pkv(batch=1, n_heads=1, seq_len=20, head_dim=4, n_layers=1)
        frames = backend._frames_for_windows(pkv=pkv, seq_len=20, model_id="m")
        assert len(frames) == 3
        assert [f[0].token_span for f in frames] == [(0, 7), (8, 15), (16, 19)]

    def test_real_persisted_to_content_store(self, monkeypatch):
        """A mocked successful forward writes frame bytes into the store."""
        from membrane.compute import kv as kv_module

        store = InProcessBytes()
        backend = kv_module.KVBackend.__new__(kv_module.KVBackend)
        backend.content_store = store
        backend.model_id = "fake-model"
        backend.model_revision = "rev1"
        backend.tokenizer_name = "fake-model"
        backend.tokenizer_revision = "rev1"
        backend.dtype = "float16"
        backend.window_size = 4
        backend.n_layers = 1
        backend.n_heads = 1
        backend.head_dim = 4

        class _FakeModel:
            config = type(
                "_C", (), {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 4}
            )()

            def __call__(self, *args, **kwargs):
                outputs = type("_O", (), {})()
                outputs.past_key_values = _make_pkv(1, 1, 4, 4, 1)
                return outputs

        class _FakeTokenizer:
            def __call__(self, text: str, **_kwargs):
                import torch as _t

                return {"input_ids": _t.ones((1, 4), dtype=_t.long)}

        backend.model = _FakeModel()
        backend.tokenizer = _FakeTokenizer()
        backend.torch = torch
        backend.actual_device = "cpu"

        frags = backend.prefill([1, 2, 3, 4], "fake-model")
        assert len(frags) == 1
        assert frags[0].payload_ref is not None
        assert store.has(frags[0].payload_ref)
        stored = store.get(frags[0].payload_ref)
        assert stored is not None
        assert len(stored) > 0


def _payload_only(frame: bytes, n_layers: int, n_heads: int, window_len: int, head_dim: int) -> bytes:
    """Drop the leading header so we can sha256 the raw K/V payload."""
    header_size = 4 * 6  # six u32s
    return frame[header_size:]
