"""OpenAIBackend: compute backend that delegates to OpenAI API.

Requires ``httpx`` (installed via ``pip install membrane[server]``).

This backend turns Membrane's compute interface into thin HTTP
calls to OpenAI's REST API:

* :meth:`prefill` — calls ``/embeddings`` to fetch a single
  embedding for the prompt and slices it across 128-token
  windows.
* :meth:`generate` — calls ``/chat/completions`` and returns
  the first choice's text.

The backend gracefully degrades when the ``httpx`` package is
missing or the API call fails: prefill falls back to a
simulation, and ``generate`` returns an empty result with a
warning.

Security:
    * The API key is passed in via ``__init__`` and never
      logged. Prefer loading it from an environment variable
      (``OPENAI_API_KEY``) rather than committing it to source.
    * The default ``base_url`` points at the public OpenAI
      endpoint; for compliance-driven deployments point it at an
      approved proxy.
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


class OpenAIBackend(ComputeBackend):
    """Compute backend using OpenAI API for embeddings and generation.

    Args:
        api_key: OpenAI API key.
        base_url: API base URL
            (default ``https://api.openai.com/v1``).
        model: Chat model name (default ``"gpt-4o-mini"``).
        embedding_model: Embedding model name
            (default ``"text-embedding-3-small"``).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the backend.

        Args:
            api_key: OpenAI API key. Stored on the instance
                but never logged.
            base_url: API base URL.
            model: Chat model identifier.
            embedding_model: Embedding model identifier.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model
        self._client: Any | None = None
        try:
            import httpx

            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0,
            )
        except ImportError:
            logger.warning("OpenAIBackend: httpx not installed")

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
        """Fetch embeddings from OpenAI and convert to fragments.

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

        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.embedding_model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            # Network/HTTP error, malformed JSON, or unexpected
            # response shape — all degrade to the simulated prefill.
            logger.warning("OpenAI embedding failed (%s); falling back to simulation", exc)
            return self.simulate_prefill(prompt_tokens, model_id)

        # Distribute the (single) embedding across windows by
        # slicing it into chunks of ``window_size`` floats (with
        # zero-padding if the embedding is shorter than the
        # chunk).
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
            "OpenAIBackend: prefill %s tokens into %s fragments",
            len(prompt_tokens),
            len(fragments),
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate text via OpenAI chat completions.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier (currently unused —
                ``self.model`` is the source of truth).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            dict: ``{"text": ..., "tokens": [...]}``. ``tokens``
            is always empty because the chat-completions API
            does not return raw token IDs. Returns empty values
            when the client is missing or the request fails.
        """
        if self._client is None:
            return {"text": "", "tokens": []}
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return {"text": choice["message"]["content"], "tokens": []}
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("OpenAI generate failed: %s", exc)
            return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return whether the API is reachable.

        Sends a lightweight ``GET /models`` request and reports
        success based on the HTTP status.

        Returns:
            bool: True when the client is configured and the
            API responds with status 200.
        """
        if self._client is None:
            return False
        try:
            resp = self._client.get(f"{self.base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def device_name(self) -> str:
        """Return the backend's device descriptor.

        Returns:
            str: ``"openai(<chat model>)"``.
        """
        return f"openai({self.model})"

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
