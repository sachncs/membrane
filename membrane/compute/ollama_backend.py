"""OllamaBackend: compute backend that delegates to a local Ollama server.

Requires ``httpx`` (installed via ``pip install membrane[server]``).
"""

import hashlib
import logging
from typing import Any

from membrane.compute.backend import ComputeBackend
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class OllamaBackend(ComputeBackend):
    """Compute backend using Ollama API for embeddings and generation.

    Args:
        base_url: Ollama server URL (default ``http://localhost:11434``).
        model: Model name to use (default ``"llama3.2"``).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Any | None = None
        try:
            import httpx
            self._client = httpx.Client(timeout=30.0)
        except ImportError:
            logger.warning("OllamaBackend: httpx not installed")

    def _hash_tokens(self, tokens: list[int]) -> str:
        payload = ",".join(str(t) for t in tokens)
        return hashlib.md5(payload.encode()).hexdigest()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Get embeddings from Ollama and convert to fragments."""
        if self._client is None:
            return self._simulate_prefill(prompt_tokens, model_id)

        # Decode tokens to text for Ollama embedding API
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
        except Exception as exc:
            logger.warning("Ollama embedding failed (%s); falling back to simulation", exc)
            return self._simulate_prefill(prompt_tokens, model_id)

        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self._hash_tokens(chunk)
            # Use a slice of the embedding for each chunk (or full embedding for first chunk)
            emb_slice = embedding[: len(chunk)] if len(embedding) >= len(chunk) else embedding + [0.0] * (len(chunk) - len(embedding))
            frag = Fragment(
                content_hash=h,
                embedding=tuple(emb_slice),
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
        logger.debug("OllamaBackend: prefill %s tokens into %s fragments", len(prompt_tokens), len(fragments))
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate text via Ollama /api/generate."""
        if self._client is None:
            return {"text": "", "tokens": []}
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": text, "stream": False, "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            data = resp.json()
            return {"text": data.get("response", ""), "tokens": []}
        except Exception as exc:
            logger.warning("Ollama generate failed: %s", exc)
            return {"text": "", "tokens": []}

    def available(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def device_name(self) -> str:
        return f"ollama({self.model})"

    def _simulate_prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
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
                size=len(chunk) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        return fragments
