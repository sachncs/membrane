"""GPU: prefill using GPU (torch CUDA) if available.

Falls back to CPU if CUDA is unavailable. This is an optional
backend that requires ``torch`` to be installed.

The backend's :meth:`__init__` probes the runtime once and
records one of three states:

* **GPU available**: ``torch`` is importable, ``cuda.is_available()``
  is ``True`` → use CUDA.
* **CPU fallback**: ``torch`` missing or CUDA unavailable →
  delegate every call to a held
  :class:`~membrane.compute.cpu_backend.CPU`.

This means callers do not need to special-case missing CUDA —
they can simply instantiate ``GPU()`` and rely on the
backend's automatic degradation.

Limitations:
    * The current implementation simulates GPU prefill by
      allocating tensors and performing a trivial reduction. It
      does not run a real model; replace with a Transformers or
      vLLM-based backend for production use.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.fragment import Fragment
from membrane.signature import Signature


class GPU(Backend):
    """GPU-based compute backend using PyTorch CUDA.

    Falls back to :class:`CPU` if ``torch`` is not
    installed or CUDA is unavailable.

    Attributes:
        _torch: Imported ``torch`` module, or ``None`` when
            unavailable.
        _device: Device string (``"cuda"``) or ``None`` when
            unavailable.
        _fallback: :class:`CPU` used when GPU is
            unavailable.
    """

    def __init__(self) -> None:
        """Probe the runtime and select GPU or CPU fallback."""
        self._torch = None
        self._device: str | None = None
        self._fallback: CPU | None = None
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                self._device = "cuda"
                logger.info("GPU: using %s", torch.cuda.get_device_name(0))
            else:
                logger.warning("GPU: CUDA unavailable, will use fallback")
                self._fallback = CPU()
        except ImportError:
            logger.warning("GPU: torch not installed, using CPU fallback")
            self._fallback = CPU()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run prefill on GPU (or fallback to CPU).

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            list[Fragment]: One fragment per window. Returns
            the CPU backend's output when GPU is unavailable.
        """
        if self._fallback is not None:
            return self._fallback.prefill(prompt_tokens, model_id)

        # GPU simulation: create tensors and simulate KV generation.
        assert self._torch is not None
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = CPU.hash_tokens(chunk)
            # Allocate a small GPU tensor for the chunk. The
            # sum() forces a synchronous kernel launch so the
            # simulator actually exercises the GPU even when no
            # real model is loaded.
            t = self._torch.tensor(chunk, device=self._device)
            _ = t.sum().item()
            frag = Fragment(
                content_hash=h,
                embedding=(float(i), float(len(chunk))),
                structural_signature=Signature(
                    model_id=model_id,
                    layer_range=(0, 1),
                    token_span=(i, min(i + window_size, len(prompt_tokens)) - 1),
                ),
                size=len(chunk) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        logger.debug(
            "GPU: prefill %s tokens into %s fragments on %s",
            len(prompt_tokens),
            len(fragments),
            self._device,
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate tokens on the GPU (or CPU fallback).

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            max_tokens: Maximum tokens to generate.

        Returns:
            dict: ``{"text": ..., "tokens": [...]}``. Falls back
            to the CPU backend's stub when GPU is unavailable.
        """
        if self._fallback is not None:
            return self._fallback.generate(prompt_tokens, model_id, max_tokens)
        return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return whether GPU prefill is usable.

        Returns:
            bool: True when the backend is using CUDA, False
            when it has fallen back to CPU.
        """
        return self._fallback is None

    def device_name(self) -> str:
        """Return the active device name.

        Returns:
            str: ``"cuda"`` when GPU is active; a descriptive
            fallback string of the form
            ``"gpu_fallback(cpu)"`` otherwise.
        """
        if self._fallback is not None:
            return f"gpu_fallback({self._fallback.device_name()})"
        return str(self._device)
