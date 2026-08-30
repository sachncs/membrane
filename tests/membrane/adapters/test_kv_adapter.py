"""Tests for the KVAdapter protocol + BaseAdapter (Phase 2.1-2.3)."""

from __future__ import annotations

import pytest

from membrane.adapters import (
    BaseAdapter,
    KVAdapter,
    KVTensor,
    LayerKV,
    ValidationResult,
)
from membrane.compat import compat_hash


def _kvtensor(n_layers: int = 4, head_dim: int = 64) -> KVTensor:
    """Build a small synthetic KVTensor for tests."""
    fingerprint = compat_hash(model_id="test", dtype="float16")
    shape = (8, 1, head_dim, 1)
    layers = tuple(
        LayerKV(
            layer_idx=i,
            k=memoryview(b"\x00" * (8 * 1 * head_dim * 2)),
            v=memoryview(b"\x00" * (8 * 1 * head_dim * 2)),
            head_range=(0, 7),
            dtype="float16",
        )
        for i in range(n_layers)
    )
    return KVTensor(
        layers=layers,
        layer_range=(0, n_layers - 1),
        head_range=(0, 7),
        token_span=(0, 0),
        shape=shape,
        fingerprint=fingerprint,
    )


class TestKVTensor:
    def test_size_bytes_handles_common_dtypes(self):
        for dtype, n_bytes in [
            ("float16", 2),
            ("bfloat16", 2),
            ("float32", 4),
            ("float64", 8),
        ]:
            k = KVTensor(
                layers=(
                    LayerKV(
                        layer_idx=0, k=memoryview(b"\x00" * 16),
                        v=memoryview(b"\x00" * 16),
                        head_range=(0, 1),
                        dtype=dtype,
                    ),
                ),
                layer_range=(0, 0),
                head_range=(0, 1),
                token_span=(0, 0),
                shape=(2, 1, 8, 1),
                fingerprint=compat_hash(model_id="d", dtype=dtype),
            )
            # 2 heads * 1 seq * 8 head_dim * 2 elements (K+V) * 2 bytes
            assert k.size_bytes == 2 * 1 * 8 * 2 * n_bytes

    def test_empty_bundle_size_zero(self):
        k = KVTensor(
            layers=(),
            layer_range=(0, 0),
            head_range=(-1, -1),
            token_span=(0, 0),
            shape=(1, 1, 1, 64),
            fingerprint=compat_hash(model_id="empty"),
        )
        assert k.size_bytes == 0


class TestValidationResult:
    def test_ok_factory(self):
        r = ValidationResult.ok()
        assert r.is_ok is True
        assert r.errors == ()

    def test_fail_factory(self):
        r = ValidationResult.fail("a", "b")
        assert r.is_ok is False
        assert r.errors == ("a", "b")


class TestBaseAdapterSerialize:
    def test_round_trip(self):
        adapter = BaseAdapter()
        k = _kvtensor(n_layers=4)
        payload = adapter.serialize(k)
        decoded = adapter.deserialize(payload)
        assert decoded.layer_range == k.layer_range
        assert decoded.head_range == k.head_range
        assert decoded.token_span == k.token_span
        # The wire format carries (n_heads, seq_len, head_dim) and
        # fills the 4th shape slot with a 64-byte element size
        # marker. The first three entries are preserved end-to-end.
        assert decoded.shape[:3] == k.shape[:3]
        assert len(decoded.layers) == len(k.layers)

    def test_bad_magic_raises(self):
        adapter = BaseAdapter()
        with pytest.raises(ValueError, match="invalid KVAdapter"):
            adapter.deserialize(b"NOPE\x00\x01\x00\x00\x00\x00")

    def test_bad_schema_raises(self):
        adapter = BaseAdapter()
        with pytest.raises(ValueError, match="unsupported KVAdapter"):
            adapter.deserialize(b"MVKV\xff\x00")


class TestBaseAdapterValidate:
    def test_empty_layers_fails(self):
        adapter = BaseAdapter()
        k = _kvtensor(n_layers=0)
        result = adapter.validate(k)
        assert result.is_ok is False
        assert any("no layers" in e for e in result.errors)

    def test_mismatched_head_range_fails(self):
        adapter = BaseAdapter()
        k = _kvtensor(n_layers=2)
        # Mutate the layer's head_range to disagree with the
        # bundle's head_range.
        bad = KVTensor(
            layers=(
                LayerKV(
                    layer_idx=0,
                    k=k.layers[0].k,
                    v=k.layers[0].v,
                    head_range=(0, 0),  # differs from bundle's (0, 7)
                    dtype="float16",
                ),
                k.layers[1],
            ),
            layer_range=k.layer_range,
            head_range=k.head_range,
            token_span=k.token_span,
            shape=k.shape,
            fingerprint=k.fingerprint,
        )
        result = adapter.validate(bad)
        assert result.is_ok is False

    def test_clean_bundle_passes(self):
        adapter = BaseAdapter()
        k = _kvtensor(n_layers=2)
        assert adapter.validate(k).is_ok is True


class TestKVAdapterProtocol:
    def test_protocol_is_runtime_checkable(self):
        from membrane.adapters import MembraneAdapter

        fake_backend = object()  # MembraneAdapter doesn't call it during isinstance
        assert isinstance(MembraneAdapter(fake_backend), KVAdapter)
