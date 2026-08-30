"""CPU: prefill simulation using CPU (numpy/torch CPU).

This module defines :class:`CPU`, the always-available
reference implementation of :class:`~membrane.compute.backend
.Backend`. It splits a prompt into fixed-size windows and
produces a content-addressable fragment per window without
loading any actual model weights.

The backend is suitable for:

* Unit tests that need deterministic, dependency-free prefill.
* CPU-only deployments where a real model is unavailable.
* Smoke-testing the rest of the Membrane pipeline.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.compute.base import Backend
from membrane.fragment import Fragment


class CPU(Backend):
    """CPU-based compute backend.

    Simulates prefill by converting prompt tokens into fragments.
    No actual model weights are loaded — this is a lightweight
    simulation suitable for testing and CPU-only deployments.
    """

    def __init__(self) -> None:
        """Initialize the backend."""
        self.initialized = True

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Simulate prefill on CPU.

        Splits the prompt into fixed-size windows and returns one
        fragment per window. Each fragment is built by the shared
        :meth:`Backend.simulate_prefill_fragment` helper.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            list[Fragment]: One fragment per window. Empty when
            ``prompt_tokens`` is empty.
        """
        window_size = Backend.SIMULATE_WINDOW_SIZE
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            fragments.append(
                Backend.simulate_prefill_fragment(
                    chunk=chunk,
                    chunk_index=i,
                    total_prompt_tokens=len(prompt_tokens),
                    model_id=model_id,
                    window_size=window_size,
                )
            )
        logger.debug(
            "CPU: prefill %s tokens into %s fragments",
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
