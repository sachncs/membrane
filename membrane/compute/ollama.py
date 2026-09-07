"""Ollama: compute backend that delegates to a local Ollama server.

Requires ``httpx`` (installed via ``pip install membrane[server]``).

The backend exposes the standard
:class:`~membrane.compute.backend.Backend` interface and
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

import json
import logging

import httpx

from membrane.compute._hash import token_hash
from membrane.compute.base import Backend
from membrane.compute.remote import RemoteLLMBackend
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity

logger = logging.getLogger(__name__)


class Ollama(RemoteLLMBackend):
    """Compute backend using Ollama API for embeddings and generation.

    Args:
        base_url: Ollama server URL
            (default ``http://localhost:11434``).
        model: Model name to use (default ``"llama3.2"``).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        """Initialize the backend."""
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = self.build_client(timeout=30.0)

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Fetch embeddings from Ollama and convert to fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier stamped on each
                fragment's identity.

        Returns:
            list[Fragment]: One fragment per 128-token window.
            Falls back to a simulation when the API call fails
            or the ``httpx`` client is unavailable.
        """
        if self.client is None:
            return self.simulate_prefill(prompt_tokens, model_id)

        # Ollama expects text, not raw token IDs; we stringify
        # the tokens with a space separator so the embedding is
        # deterministic for a given token sequence.
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("Ollama embedding failed (%s); falling back to simulation", exc)
            return self.simulate_prefill(prompt_tokens, model_id)

        # Distribute the embedding across 128-token windows
        # (with zero-padding when shorter). The ``emb_slice`` value
        # is no longer carried on the Fragment — the new schema
        # drops ``embedding`` — so we only log here.
        del embedding
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = token_hash(chunk)
            identity = PayloadIdentity(
                payload_hash=h,
                model_id=model_id,
                model_revision="",
                tokenizer_name=model_id,
                tokenizer_revision="",
                layer_range=(0, 1),
                head_range=(-1, -1),
                token_span=(i, min(i + window_size, len(prompt_tokens)) - 1),
                dtype="float16",
                shape=(1, 1, len(chunk), 1, 64),
            )
            frag = Fragment(
                identity=identity,
                payload_ref=h,
                payload_size=len(chunk) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        logger.debug(
            "Ollama: prefill %s tokens into %s fragments",
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
        if self.client is None:
            return {"text": "", "tokens": []}
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self.client.post(
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
        """Return whether Ollama is reachable."""
        return self.probe("api/tags")

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
        """Simulated prefill used when the API call is unavailable."""
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
        return fragments
