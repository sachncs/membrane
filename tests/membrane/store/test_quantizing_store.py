"""Quantization at rest: smaller tensors, faithful round trips, pass-through for the rest."""

import numpy as np
import pytest

from membrane.canonical import canonicalize, parse_canonical
from membrane.content_store import InProcessBytes
from membrane.identity import PayloadIdentity
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.store.quantizing import MAGIC, QuantizingStore


def tensor_frame(shape=(2, 4, 16, 64), dtype="float16") -> tuple[bytes, np.ndarray, PayloadIdentity]:
    rng = np.random.default_rng(7)
    tensor = rng.standard_normal(shape).astype(dtype)
    identity = PayloadIdentity(
        payload_hash="q" * 64,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(0, 3),
        token_span=(0, 15),
        dtype=dtype,
        shape=shape,
    )
    return canonicalize(identity, tensor.tobytes()), tensor, identity


@pytest.mark.parametrize(("fmt", "max_error"), [("int8", 0.05), ("fp8_e4m3", 0.2), ("nf4", 0.6)])
def test_tensor_frames_round_trip_within_tolerance(fmt: str, max_error: float) -> None:
    frame, tensor, identity = tensor_frame()
    store = QuantizingStore(InProcessBytes(), fmt)
    store.put("k", frame)
    stored = store.inner.get("k")
    assert stored.startswith(MAGIC) and len(stored) < len(frame)
    got_identity, raw = parse_canonical(store.get("k"))
    assert got_identity == identity
    restored = np.frombuffer(raw, dtype="float16").reshape(tensor.shape)
    assert float(np.max(np.abs(restored.astype("float32") - tensor.astype("float32")))) < max_error * float(
        np.max(np.abs(tensor))
    )


def test_int8_halves_float16_payloads() -> None:
    frame, _, _ = tensor_frame()
    store = QuantizingStore(InProcessBytes(), "int8")
    store.put("k", frame)
    assert store.size() < 0.6 * len(frame)


def test_non_tensor_payloads_pass_through() -> None:
    store = QuantizingStore(InProcessBytes(), "int8")
    store.put("raw", b"simulated placeholder bytes")
    assert store.get("raw") == b"simulated placeholder bytes"
    assert store.inner.get("raw") == b"simulated placeholder bytes"
    assert store.has("raw") and store.delete("raw") and store.get("raw") is None


def test_server_wraps_the_content_store() -> None:
    server, _ = build_server(ServerSettings(port=0, kv_quantization="int8", load_hooks=False))
    assert isinstance(server.node.content_store, QuantizingStore)
    with pytest.raises(SettingsError, match="KV quantization must be"):
        ServerSettings(kv_quantization="int2")
