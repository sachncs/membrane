"""OllamaBackend: compute backend that delegates to a local Ollama server.

Requires ``httpx`` (installed via ``pip install membrane[server]``).

The backend exposes the standard
:class:`~membrane.compute.backend.ComputeBackend` interface and
backs it with HTTP calls to a locally running Ollama daemon:

* :meth:`prefill` — calls ``POST /api/embeddings`` to fetch a
  prompt embedding and slices it across 128-token windows.
* :meth:`generate` — calls ``POST /api/generate`` with
  ``stream=False`` and returns the produced text.
* :meth:`available` — calls ``GET /api/tags`` as a cheap
  liveness probe.

The backend gracefully degrades when the ``httpx`` package is
missing or the API call fails: prefill falls back to a small
simulation, and ``generate`` returns an empty result with a
warning.
"""

import hashlib
import json
import logging
from typing import Any

import httpx

from membrane.compute.backend import ComputeBackend
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class OllamaBackend(ComputeBackend):
    """Compute backend using Ollama API for embeddings and generation.

    Args:
        base_url: Ollama server URL
            (default ``http://localhost:11434``).
        model: Model name to use (default ``"llama3.2"``).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        """Initialize the backend.

        Args:
            base_url: Ollama server URL.
            model: Model identifier.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Any | None = None
        try:
            import httpx
            self._client = httpx.Client(timeout=30.0)
        except ImportError:
            logger.warning("OllamaBackend: httpx not installed")

    def hash_tokens(self, tokens: list[int]) -> str:
        """MD5-hash a token chunk.

        Args:
            tokens: Token IDs.

        Returns:
            str: Hexadecimal digest.
        """
        payload = ",".join(str(t) for t in tokens)
        return hashlib.md5(payload.encode()).hexdigest()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Fetch embeddings from Ollama and convert to fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier stamped on each
                fragment's structural signature.

        Returns:
            list[Fragment]: One fragment per 128-token window.
            Falls back to a simulation when the API call fails
            or the ``httpx`` client is unavailable.
        """
        if self._client is None:
            return self.simulate_prefill(prompt_tokens, model_id)

        # Ollama expects text, not raw token IDs; we stringify
        # the tokens with a space separator so the embedding is
        # deterministic for a given token sequence.
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning(
                "Ollama embedding failed (%s); falling back to simulation", exc
            )
            return self.simulate_prefill(prompt_tokens, model_id)

        # Distribute the embedding across 128-token windows
        # (with zero-padding when shorter).
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self.hash_tokens(chunk)
            emb_slice = (
                embedding[: len(chunk)]
                if len(embedding) >= len(chunk)
                else embedding + [0.0] * (len(chunk) - len(embedding))
            )
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
        logger.debug(
            "OllamaBackend: prefill %s tokens into %s fragments",
            len(prompt_tokens),
            len(fragments),
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate text via Ollama's ``/api/generate``.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier (currently unused —
                ``self.model`` is the source of truth).
            max_tokens: Maximum tokens to generate, passed as
                Ollama's ``num_predict`` option.

        Returns:
            dict: ``{"text": ..., "tokens": [...]}``. ``tokens``
            is always empty because the API does not return raw
            token IDs. Empty values when the client is missing
            or the request fails.
        """
        if self._client is None:
            return {"text": "", "tokens": []}
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": text,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {"text": data.get("response", ""), "tokens": []}
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("Ollama generate failed: %s", exc)
            return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return whether Ollama is reachable.

        Returns:
            bool: True when the client is configured and the
            server responds with status 200 to ``GET /api/tags``.
        """
        if self._client is None:
            return False
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def device_name(self) -> str:
        """Return the backend's device descriptor.

        Returns:
            str: ``"ollama(<model>)"``.
        """
        return f"ollama({self.model})"

    def simulate_prefill(
        self,
        prompt_tokens: list[int],
        model_id: str,
    ) -> list[Fragment]:
        """Simulated prefill used when the API call is unavailable.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier for the structural
                signature.

        Returns:
            list[Fragment]: One fragment per 128-token window
            with a placeholder embedding.
        """
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self.hash_tokens(chunk)
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
