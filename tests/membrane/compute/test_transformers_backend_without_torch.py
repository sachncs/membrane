"""The Transformers backend's model paths, run against numpy stand-ins for torch."""

from membrane.compute.transformers import Transformers
from tests.membrane.compute.fake_torch import Model, install


def test_loads_on_cpu_and_prefills_one_fragment_per_window(monkeypatch) -> None:
    install(monkeypatch, Model())
    backend = Transformers(model_id="tiny")
    assert backend.available()
    assert backend.device_name() == "transformers(tiny,cpu)"
    fragments = backend.prefill(list(range(130)), "tiny")
    assert [f.identity.token_span for f in fragments] == [(0, 127), (128, 129)]
    assert fragments[0].identity.model_id == "tiny"


def test_generate_returns_only_new_tokens(monkeypatch) -> None:
    install(monkeypatch, Model())
    backend = Transformers(model_id="tiny", device="cpu")
    result = backend.generate([1, 2, 3], "tiny", max_tokens=2)
    assert result == {"text": "7 7", "tokens": [7, 7]}


def test_model_failures_degrade(monkeypatch) -> None:
    install(monkeypatch, Model(fail=True))
    backend = Transformers(model_id="tiny")
    assert len(backend.prefill([1, 2, 3], "m")) == 1  # simulated
    assert backend.generate([1], "m") == {"text": "", "tokens": []}


def test_load_errors_leave_it_unloaded(monkeypatch) -> None:
    install(monkeypatch, Model(), load_error=OSError("offline"))
    backend = Transformers(model_id="tiny")
    assert not backend.available()
    assert backend.device_name() == "transformers(unloaded)"
    assert backend.generate([1], "m") == {"text": "", "tokens": []}
