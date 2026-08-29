"""Compute backends for Membrane prefill and inference.

This package groups the compute backend implementations that
power Membrane prefill and decode operations. Each backend
exposes the same :class:`~membrane.compute.backend.Backend`
protocol so callers can swap backends without touching their
code.

Available backends:

* :class:`~membrane.compute.cpu.CPU` — pure-Python
  reference implementation; always available.
* :class:`~membrane.compute.gpu.GPU` — PyTorch
  CUDA backend; requires the ``[gpu]`` extra.
* :class:`~membrane.compute.transformers.Transformers`
  — HuggingFace Transformers backend; requires the
  ``[local-llm]`` extra.
* :class:`~membrane.compute.openai.OpenAI` —
  OpenAI API backend; requires the ``openai`` package.
* :class:`~membrane.compute.anthropic.Anthropic`
  — Anthropic API backend; requires the ``anthropic`` package.
* :class:`~membrane.compute.ollama.Ollama` —
  Ollama local server backend.

Optional backends are imported lazily and listed in
``__all__`` only when their optional dependency is available,
keeping ``import membrane.compute`` fast in minimal
installations.
"""

from membrane.compute.base import Backend
from membrane.compute.cpu import CPU

__all__ = ["Backend", "CPU"]


def try_register(name: str, module_path: str) -> None:
    """Attempt to import an optional backend and add it to ``__all__``.

    Args:
        name: Public class name to register.
        module_path: Dotted module path to import from.
    """
    try:
        __import__(module_path, fromlist=[name])
        __all__.append(name)
    except ImportError:
        # The optional dependency is not installed; skip silently
        # so that ``import membrane.compute`` continues to work
        # in minimal environments.
        pass


for _backend_name, _backend_path in (
    ("GPU", "membrane.compute.gpu"),
    ("Ollama", "membrane.compute.ollama"),
    ("OpenAI", "membrane.compute.openai"),
    ("Anthropic", "membrane.compute.anthropic"),
    ("Transformers", "membrane.compute.transformers"),
):
    try_register(_backend_name, _backend_path)
