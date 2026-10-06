"""The GPU backend with CUDA present and absent, against a stand-in torch module."""

import sys
import types

import pytest

from membrane.compute.gpu import GPU


def fake_torch(monkeypatch, cuda: bool) -> list[list[int]]:
    seen: list[list[int]] = []

    class Tensor:
        def __init__(self, values, device=None):
            self.values = list(values)
            self.device = device
            seen.append(self.values)

        def sum(self):
            return types.SimpleNamespace(item=lambda: sum(self.values))

    torch = types.ModuleType("torch")
    torch.tensor = Tensor
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda index: "Fake GPU")
    monkeypatch.setitem(sys.modules, "torch", torch)
    return seen


def test_cuda_prefill_runs_each_window_on_the_device(monkeypatch) -> None:
    seen = fake_torch(monkeypatch, cuda=True)
    backend = GPU()
    assert backend.available() and backend.device_name() == "cuda"
    fragments = backend.prefill(list(range(130)), "m")
    assert len(fragments) == 2 and len(seen) == 2
    assert backend.generate([1], "m") == {"text": "", "tokens": []}


def test_without_cuda_it_delegates_to_cpu(monkeypatch) -> None:
    fake_torch(monkeypatch, cuda=False)
    backend = GPU()
    assert not backend.available()
    assert backend.device_name().startswith("gpu_fallback")
    assert len(backend.prefill([1, 2, 3], "m")) == 1
    assert isinstance(backend.generate([1, 2], "m"), dict)


def test_without_torch_it_delegates_to_cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    backend = GPU()
    assert backend.fallback is not None


@pytest.mark.parametrize("cuda", [True, False])
def test_device_name_is_stable(monkeypatch, cuda: bool) -> None:
    fake_torch(monkeypatch, cuda=cuda)
    assert GPU().device_name() == GPU().device_name()
