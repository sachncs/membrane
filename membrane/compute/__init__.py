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

try:
    from membrane.compute.gpu_backend import GPUBackend
    __all__.append("GPUBackend")
except ImportError:
    pass

try:
    from membrane.compute.ollama_backend import OllamaBackend
    __all__.append("OllamaBackend")
except ImportError:
    pass

try:
    from membrane.compute.openai_backend import OpenAIBackend
    __all__.append("OpenAIBackend")
except ImportError:
    pass

try:
    from membrane.compute.anthropic_backend import AnthropicBackend
    __all__.append("AnthropicBackend")
except ImportError:
    pass

try:
    from membrane.compute.transformers_backend import TransformersBackend
    __all__.append("TransformersBackend")
except ImportError:
    pass
