"""Backend: abstraction for prefill/inference compute.

This module defines :class:`Backend`, the abstract base
class every concrete compute backend must implement. The
interface is intentionally minimal: prefill, generate, and two
metadata accessors (availability and device name).

All methods are described as "asynchronous-friendly" — they
should avoid long blocking operations when called from an event
loop. Concrete backends that perform CPU- or GPU-bound work
typically offload to a thread or process pool.

The base class also provides a shared
:func:`simulate_prefill_fragment` static helper used as the
fallback path when a real backend (OpenAI / Ollama /
Transformers) cannot reach its provider. Every backend used
to inline the same boilerplate; consolidating it here removes
three copies and ensures the field defaults (``ttl``,
``reuse_score``, ``version_id``) stay consistent across
backends.
"""

from abc import ABC, abstractmethod

from membrane.compute._hash import token_hash
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


class Backend(ABC):
    """Abstract compute backend for KV-cache prefill and decode.

    Implementations may use CPU (numpy/torch CPU) or GPU (CUDA).
    All methods are asynchronous-friendly (non-blocking).

    Concrete subclasses are expected to be safe to instantiate
    at process start; expensive resources (e.g., model weights)
    should be loaded lazily on first use.
    """

    #: Default window size used by :meth:`simulate_prefill`.
    SIMULATE_WINDOW_SIZE: int = 128

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

    @staticmethod
    def simulate_prefill_fragment(
        chunk: list[int],
        chunk_index: int,
        total_prompt_tokens: int,
        model_id: str,
        window_size: int = 128,
    ) -> Fragment:
        """Build a placeholder :class:`Fragment` for a single simulated window.

        Shared helper for every backend's fallback path. The
        returned fragment uses ``token_hash`` for content
        addressing and stamps a synthetic layer/head/token span
        on the :class:`~membrane.identity.PayloadIdentity` so
        the chunk is recoverable from the fragment alone.

        Args:
            chunk: Token IDs for this window.
            chunk_index: Offset of the chunk in the original
                prompt (in tokens).
            total_prompt_tokens: Total prompt size, used to
                clip the final window's ``token_span``.
            model_id: Model identifier stamped on the
                fragment's identity.
            window_size: Window size that produced this chunk.

        Returns:
            Fragment: Placeholder fragment with deterministic
            content hash and synthetic embedding.
        """
        payload_hash = token_hash(chunk)
        token_span_end = min(chunk_index + window_size, total_prompt_tokens) - 1
        identity = PayloadIdentity(
            payload_hash=payload_hash,
            model_id=model_id,
            model_revision="",
            tokenizer_name=model_id,
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(chunk_index, token_span_end),
            dtype="float16",
            shape=(1, 1, len(chunk), 1, 64),
        )
        return Fragment(
            identity=identity,
            payload_ref=payload_hash,
            payload_size=len(chunk) * 64,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )


__all__ = ["Backend"]
