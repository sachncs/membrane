"""CPUBackend: prefill simulation using CPU (numpy/torch CPU)."""

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
        self._initialized = True

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Simulate prefill on CPU.

        Splits the prompt into fixed-size windows and returns one
        Fragment per window.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            List of fragments.
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
                size=len(chunk) * 64,  # rough bytes per token
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        logger.debug("CPUBackend: prefill %s tokens into %s fragments", len(prompt_tokens), len(fragments))
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        return {"text": "", "tokens": []}

    def available(self) -> bool:
        return True

    def device_name(self) -> str:
        return "cpu"

    @staticmethod
    def _hash_tokens(tokens: list[int]) -> str:
        import hashlib

        payload = ",".join(str(t) for t in tokens)
        return hashlib.md5(payload.encode()).hexdigest()
