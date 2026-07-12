"""ComputeBackend: abstraction for prefill/inference compute.

This module defines :class:`ComputeBackend`, the abstract base
class every concrete compute backend must implement. The
interface is intentionally minimal: prefill, generate, and two
metadata accessors (availability and device name).

All methods are described as "asynchronous-friendly" — they
should avoid long blocking operations when called from an event
loop. Concrete backends that perform CPU- or GPU-bound work
typically offload to a thread or process pool.
"""

from abc import ABC, abstractmethod

from membrane.fragment import Fragment


class ComputeBackend(ABC):
    """Abstract compute backend for KV-cache prefill and decode.

    Implementations may use CPU (numpy/torch CPU) or GPU (CUDA).
    All methods are asynchronous-friendly (non-blocking).

    Concrete subclasses are expected to be safe to instantiate
    at process start; expensive resources (e.g., model weights)
    should be loaded lazily on first use.
    """

    @abstractmethod
    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run prefill on a prompt and return fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            list[Fragment]: Fragments representing the KV cache.
        """

    @abstractmethod
    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Run text generation on a prompt.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            max_tokens: Maximum tokens to generate.

        Returns:
            dict: Result with at least ``text`` and ``tokens``
            keys.
        """

    @abstractmethod
    def available(self) -> bool:
        """Return whether this backend is available on the current host.

        Returns:
            bool: True when the underlying runtime (CUDA,
            Transformers, remote API credentials, etc.) is
            reachable.
        """

    @abstractmethod
    def device_name(self) -> str:
        """Return human-readable device name (e.g., ``"cpu"`` or ``"cuda:0"``).

        Returns:
            str: Device name.
        """
