"""KVBackend frames real K/V tensors per window, and falls back to simulation.

``torch`` and ``transformers`` are optional, so the model is faked with
numpy arrays behind the small tensor surface the backend uses.
"""

import hashlib
import struct

import numpy as np
import pytest

from membrane.compute.base import Backend
from membrane.compute.kv import KVBackend
from membrane.content_store import InProcessBytes
from tests.membrane.compute.fake_torch import HEAD_DIM, HEADS, LAYERS, Model, install


def test_prefill_frames_each_window_of_real_kv(monkeypatch) -> None:
    install(monkeypatch, Model())
    store = InProcessBytes()
    backend = KVBackend(content_store=store, model_id="tiny", dtype="float16", window_size=3)
    assert backend.available() and backend.device_name() == "kv(tiny,float16,cpu)"
    fragments = backend.prefill(list(range(7)), "tiny")
    assert [f.identity.token_span for f in fragments] == [(0, 2), (3, 5), (6, 6)]
    first = fragments[0]
    frame = store.get(first.payload_ref)
    header = struct.unpack("<IIIIII", frame[:24])
    assert header == (2, LAYERS, HEADS, 3, HEAD_DIM, 2)
    payload = frame[24:]
    assert first.identity.payload_hash == hashlib.sha256(payload).hexdigest()
    assert first.identity.shape == (1, LAYERS, HEADS, 3, HEAD_DIM)
    # K of layer 0 first, cast to float16.
    k0 = np.frombuffer(payload[: HEADS * 3 * HEAD_DIM * 2], dtype=np.float16)
    assert set(k0.tolist()) == {0.0}
    # Real frames are already stored: no placeholder bytes.
    assert backend.simulated_payload(first) is None
    assert backend.generate([1], "tiny") == {"text": "", "tokens": []}


def test_bind_redirects_frames(monkeypatch) -> None:
    install(monkeypatch, Model())
    backend = KVBackend(content_store=InProcessBytes(), window_size=4)
    node_store = InProcessBytes()
    backend.bind(node_store)
    fragments = backend.prefill([1, 2, 3], "m")
    assert node_store.has(fragments[0].payload_ref)


@pytest.mark.parametrize("model", [Model(fail=True), Model(empty=True)])
def test_forward_failures_fall_back_to_simulation(monkeypatch, model: Model) -> None:
    install(monkeypatch, model)
    backend = KVBackend(content_store=InProcessBytes())
    fragments = backend.prefill(list(range(Backend.SIMULATE_WINDOW_SIZE + 1)), "m")
    assert len(fragments) == 2
    assert backend.simulated_payload(fragments[0]) is not None


def test_load_failure_leaves_the_backend_simulating(monkeypatch) -> None:
    install(monkeypatch, Model(), load_error=OSError("no weights"))
    backend = KVBackend(content_store=InProcessBytes())
    assert not backend.available()
    assert backend.device_name() == "kv(gpt2,float16,unloaded)"
    assert len(backend.prefill([1, 2], "m")) == 1


def test_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        KVBackend(content_store=InProcessBytes(), dtype="int3")
