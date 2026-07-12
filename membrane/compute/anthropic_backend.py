"""AnthropicBackend: compute backend that delegates to Anthropic API.

Requires ``httpx`` (installed via ``pip install membrane[server]``).
Anthropic has no public embedding endpoint, so prefill falls
back to hashing the prompt text.

The backend wraps Anthropic's ``/messages`` endpoint and exposes
the standard :class:`~membrane.compute.backend.ComputeBackend`
interface:

* :meth:`prefill` — produces content-addressed fragments using a
  deterministic hash of the prompt (since Anthropic does not
  expose embeddings).
* :meth:`generate` — sends the prompt as a user message and
  returns the first text block from the response.

Security:
    * The API key is stored on the instance but never logged.
      Prefer loading it from an environment variable
      (``ANTHROPIC_API_KEY``).
    * The default ``base_url`` points at the public Anthropic
      endpoint; for compliance-driven deployments point it at an
      approved proxy.
"""

import hashlib
import logging
from typing import Any

from membrane.compute.backend import ComputeBackend
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class AnthropicBackend(ComputeBackend):
    """Compute backend using Anthropic API for generation.

    Args:
        api_key: Anthropic API key.
        base_url: API base URL
            (default ``https://api.anthropic.com/v1``).
        model: Model name
            (default ``"claude-3-sonnet-20240229"``).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        model: str = "claude-3-sonnet-20240229",
    ) -> None:
        """Initialize the backend.

        Args:
            api_key: Anthropic API key.
            base_url: API base URL.
            model: Model identifier.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Any | None = None
        try:
            import httpx
            self._client = httpx.Client(
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=60.0,
            )
        except ImportError:
            logger.warning("AnthropicBackend: httpx not installed")

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
        """Produce content-addressed fragments by hashing the prompt.

        Anthropic does not currently expose an embeddings
        endpoint, so this backend derives the embedding as a
        deterministic placeholder. The fragments remain useful
        for content-addressed lookups and cache keying — they
        just do not carry real semantic embeddings.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier for the structural
                signature.

        Returns:
            list[Fragment]: One fragment per 128-token window.
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
        logger.debug(
            "AnthropicBackend: prefill %s tokens into %s fragments",
            len(prompt_tokens),
            len(fragments),
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate text via the Anthropic Messages API.

        Sends the prompt as a user message; concatenates every
        text block in the response into the returned ``text``
        field. ``tokens`` is always empty because the API does
        not expose raw token IDs.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier (currently unused —
                ``self.model`` is the source of truth).
            max_tokens: Maximum tokens to generate.

        Returns:
            dict: ``{"text": ..., "tokens": [...]}``. Empty
            values when the client is missing or the request
            fails.
        """
        if self._client is None:
            return {"text": "", "tokens": []}
        text = " ".join(str(t) for t in prompt_tokens)
        try:
            resp = self._client.post(
                f"{self.base_url}/messages",
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": text}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            # Concatenate every text block in order; non-text
            # blocks are skipped.
            text_out = ""
            for block in content:
                if block.get("type") == "text":
                    text_out += block.get("text", "")
            return {"text": text_out, "tokens": []}
        except Exception as exc:
            logger.warning("Anthropic generate failed: %s", exc)
            return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return whether the API is reachable.

        Returns:
            bool: True when the client is configured and the
            API responds with status 200 to ``GET /models``.
        """
        if self._client is None:
            return False
        try:
            resp = self._client.get(f"{self.base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def device_name(self) -> str:
        """Return the backend's device descriptor.

        Returns:
            str: ``"anthropic(<model>)"``.
        """
        return f"anthropic({self.model})"
