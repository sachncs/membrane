"""GPUBackend: prefill using GPU (torch CUDA) if available.

Falls back to CPU if CUDA is unavailable.  This is an optional backend
that requires ``torch`` to be installed.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.compute.backend import ComputeBackend
from membrane.compute.cpu_backend import CPUBackend
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


class GPUBackend(ComputeBackend):
    """GPU-based compute backend using PyTorch CUDA.

    Falls back to CPUBackend if torch is not installed or CUDA
    is unavailable.
    """

    def __init__(self) -> None:
        self._torch = None
        self._device: str | None = None
        self._fallback: CPUBackend | None = None
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                self._device = "cuda"
                logger.info("GPUBackend: using %s", torch.cuda.get_device_name(0))
            else:
                logger.warning("GPUBackend: CUDA unavailable, will use fallback")
                self._fallback = CPUBackend()
        except ImportError:
            logger.warning("GPUBackend: torch not installed, using CPU fallback")
            self._fallback = CPUBackend()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run prefill on GPU (or fallback to CPU).

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            List of fragments.
        """
        if self._fallback is not None:
            return self._fallback.prefill(prompt_tokens, model_id)

        assert self._torch is not None

        # GPU simulation: create tensors and simulate KV generation
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = CPUBackend._hash_tokens(chunk)
            # Simulate GPU work by allocating a small tensor
            t = self._torch.tensor(chunk, device=self._device)
            _ = t.sum().item()  # tiny compute to ensure GPU sync
            frag = Fragment(
                content_hash=h,
                embedding=(float(i), float(len(chunk))),
                structural_signature=StructuralSignature(
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
            "GPUBackend: prefill %s tokens into %s fragments on %s",
            len(prompt_tokens),
            len(fragments),
            self._device,
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        if self._fallback is not None:
            return self._fallback.generate(prompt_tokens, model_id, max_tokens)
        return {"text": "", "tokens": []}

    def available(self) -> bool:
        return self._fallback is None

    def device_name(self) -> str:
        if self._fallback is not None:
            return f"gpu_fallback({self._fallback.device_name()})"
        return str(self._device)
