"""Compute backends for Membrane prefill and inference.

This package groups the compute backend implementations that
power Membrane prefill and decode operations. Each backend
exposes the same :class:`~membrane.compute.backend.ComputeBackend`
protocol so callers can swap backends without touching their
code.

Available backends:

* :class:`~membrane.compute.cpu_backend.CPUBackend` — pure-Python
  reference implementation; always available.
* :class:`~membrane.compute.gpu_backend.GPUBackend` — PyTorch
  CUDA backend; requires the ``[gpu]`` extra.
* :class:`~membrane.compute.transformers_backend.TransformersBackend`
  — HuggingFace Transformers backend; requires the
  ``[local-llm]`` extra.
* :class:`~membrane.compute.openai_backend.OpenAIBackend` —
  OpenAI API backend; requires the ``openai`` package.
* :class:`~membrane.compute.anthropic_backend.AnthropicBackend`
  — Anthropic API backend; requires the ``anthropic`` package.
* :class:`~membrane.compute.ollama_backend.OllamaBackend` —
  Ollama local server backend.

Optional backends are imported lazily and listed in
``__all__`` only when their optional dependency is available,
keeping ``import membrane.compute`` fast in minimal
installations.
"""

from membrane.compute.backend import ComputeBackend
from membrane.compute.cpu_backend import CPUBackend

__all__ = ["ComputeBackend", "CPUBackend"]


def _try_register(name: str, module_path: str) -> None:
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
    ("GPUBackend", "membrane.compute.gpu_backend"),
    ("OllamaBackend", "membrane.compute.ollama_backend"),
    ("OpenAIBackend", "membrane.compute.openai_backend"),
    ("AnthropicBackend", "membrane.compute.anthropic_backend"),
    ("TransformersBackend", "membrane.compute.transformers_backend"),
):
    _try_register(_backend_name, _backend_path)
