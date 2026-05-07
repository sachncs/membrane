"""Compute backends for Membrane prefill and inference."""

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
