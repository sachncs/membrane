"""Stand-ins for ``torch`` and ``transformers`` (optional extras) backed by numpy.

They implement only the tensor and model surface the compute backends call,
so the backends' real code paths run without the extras installed.
"""

import sys
import types

import numpy as np

LAYERS, HEADS, HEAD_DIM, HIDDEN = 2, 2, 4, 8


class Tensor:
    """The slice of the torch.Tensor API the backends call."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self, index):
        return Tensor(self.array[index])

    def contiguous(self) -> Tensor:
        return Tensor(np.ascontiguousarray(self.array))

    def cpu(self) -> Tensor:
        return self

    def to(self, target):
        if isinstance(target, (type, np.dtype)):
            return Tensor(self.array.astype(target))
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def tolist(self) -> list:
        return self.array.tolist()


class Model:
    """A causal LM: K/V caches, hidden states, and greedy generation."""

    def __init__(self, fail: bool = False, empty: bool = False) -> None:
        self.config = types.SimpleNamespace(num_hidden_layers=LAYERS, num_attention_heads=HEADS, hidden_size=HIDDEN)
        self.fail = fail
        self.empty = empty
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, **kwargs):
        if self.fail:
            raise RuntimeError("forward failed")
        seq = input_ids.shape[1]
        pkv = [
            (
                Tensor(np.full((1, HEADS, seq, HEAD_DIM), layer, dtype=np.float32)),
                Tensor(np.full((1, HEADS, seq, HEAD_DIM), layer + 10, dtype=np.float32)),
            )
            for layer in range(LAYERS)
        ]
        hidden = Tensor(np.arange(seq * HIDDEN, dtype=np.float32).reshape(1, seq, HIDDEN))
        return types.SimpleNamespace(past_key_values=[] if self.empty else pkv, hidden_states=[hidden, hidden])

    def generate(self, input_ids=None, max_new_tokens=1, **kwargs):
        if self.fail:
            raise RuntimeError("generate failed")
        prompt = input_ids.array[0].tolist()
        return Tensor(np.array([prompt + [7] * max_new_tokens]))


class Tokenizer:
    """Whitespace tokenizer returning ``input_ids`` tensors."""

    def __call__(self, text, **kwargs):
        return {"input_ids": Tensor(np.arange(len(text.split()), dtype=np.int64).reshape(1, -1))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids.tolist())


def install(monkeypatch, model: Model, load_error: Exception | None = None) -> None:
    """Make ``import torch`` and ``import transformers`` return the fakes.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        model: Returned by every ``from_pretrained``.
        load_error: Raised by ``from_pretrained`` instead, when given.
    """
    torch = types.ModuleType("torch")
    torch.float16 = np.float16
    torch.float32 = np.float32
    torch.bfloat16 = np.float32
    torch.float64 = np.float64
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.no_grad = NoGrad

    def load(*args, **kwargs):
        if load_error is not None:
            raise load_error
        return model

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load)
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=load)
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
