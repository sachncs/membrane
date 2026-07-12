"""CPUBackend: prefill simulation using CPU (numpy/torch CPU).

This module defines :class:`CPUBackend`, the always-available
reference implementation of :class:`~membrane.compute.backend
.ComputeBackend`. It splits a prompt into fixed-size windows and
produces a content-addressable fragment per window without
loading any actual model weights.

The backend is suitable for:

* Unit tests that need deterministic, dependency-free prefill.
* CPU-only deployments where a real model is unavailable.
* Smoke-testing the rest of the Membrane pipeline.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.compute.backend import ComputeBackend
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


class CPUBackend(ComputeBackend):
    """CPU-based compute backend.

    Simulates prefill by converting prompt tokens into fragments.
    No actual model weights are loaded — this is a lightweight
    simulation suitable for testing and CPU-only deployments.
    """

    def __init__(self) -> None:
        """Initialize the backend.

        The flag ``_initialized`` is kept for symmetry with
        backends that lazy-load heavy resources; here it is
        always ``True``.
        """
        self._initialized = True

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Simulate prefill on CPU.

        Splits the prompt into fixed-size windows and returns one
        fragment per window. The embedding is a simple
        ``(start_offset, chunk_length)`` tuple that uniquely
        identifies the chunk's position in the prompt.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            list[Fragment]: One fragment per window. Empty when
            ``prompt_tokens`` is empty.
        """
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self._hash_tokens(chunk)
            frag = Fragment(
                content_hash=h,
                embedding=(float(i), float(len(chunk))),
                structural_signature=StructuralSignature(
                    model_id=model_id,
                    layer_range=(0, 1),
                    token_span=(i, min(i + window_size, len(prompt_tokens)) - 1),
                ),
                # Rough bytes-per-token estimate used for capacity
                # accounting rather than transport.
                size=len(chunk) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        logger.debug(
            "CPUBackend: prefill %s tokens into %s fragments",
            len(prompt_tokens),
            len(fragments),
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Stub text-generation entry point.

        Args:
            prompt_tokens: Input token IDs (unused by the stub).
            model_id: Model identifier (unused by the stub).
            max_tokens: Maximum tokens to generate (unused).

        Returns:
            dict: ``{"text": "", "tokens": []}``. The CPU backend
            is a prefill simulator and does not produce output
            tokens.
        """
        return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return availability.

        Returns:
            bool: Always ``True`` for the CPU backend.
        """
        return True

    def device_name(self) -> str:
        """Return device name.

        Returns:
            str: Always ``"cpu"``.
        """
        return "cpu"

    @staticmethod
    def _hash_tokens(tokens: list[int]) -> str:
        """Compute a deterministic MD5 digest over a token chunk.

        Args:
            tokens: Token IDs to hash.

        Returns:
            str: Hexadecimal MD5 digest.
        """
        import hashlib

        payload = ",".join(str(t) for t in tokens)
        return hashlib.md5(payload.encode()).hexdigest()
