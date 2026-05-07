"""ComputeBackend: abstraction for prefill/inference compute."""

from abc import ABC, abstractmethod

from membrane.fragment import Fragment


class ComputeBackend(ABC):
    """Abstract compute backend for KV-cache prefill and decode.

    Implementations may use CPU (numpy/torch CPU) or GPU (CUDA).
    All methods are asynchronous-friendly (non-blocking).
    """

    @abstractmethod
    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run prefill on a prompt and return fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            List of fragments representing the KV cache.
        """

    @abstractmethod
    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Run text generation on a prompt.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with at least ``text`` and ``tokens`` keys.
        """

    @abstractmethod
    def available(self) -> bool:
        """Return whether this backend is available on the current host."""

    @abstractmethod
    def device_name(self) -> str:
        """Return human-readable device name (e.g. ``"cpu"`` or ``"cuda:0"``)."""
