"""RemoteLLMBackend: shared base for HTTP-based LLM providers.

OpenAI, Anthropic, and Ollama all construct an ``httpx.Client``
with slightly different headers / timeouts, expose liveness
checks via a tiny GET probe, and degrade gracefully when the
client is missing. The duplication is large enough to factor
out, so every remote backend inherits from
:class:`RemoteLLMBackend` and only supplies the provider-
specific URL paths and request shapes.

Subclasses are expected to:

* override :meth:`device_name` for their own display string,
* call :meth:`build_client` from ``__init__`` to construct the
  shared ``httpx.Client`` with the right headers/timeouts,
* use :attr:`client` (and :meth:`available`) for runtime
  checks.
"""

from __future__ import annotations

import logging
from typing import Any

from membrane.compute.base import Backend

logger = logging.getLogger(__name__)


class RemoteLLMBackend(Backend):
    """Base class for HTTP-based LLM providers.

    Attributes:
        client: ``httpx.Client`` instance, or ``None`` when the
            dependency isn't installed / construction failed.
    """

    #: Base URL the subclass must set (e.g.,
    #: ``"https://api.openai.com/v1"`` or
    #: ``"http://localhost:11434"``).
    base_url: str = ""

    def __init__(self) -> None:
        self.client: Any | None = None

    def build_client(
        self,
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Construct an ``httpx.Client`` with the supplied options.

        Returns ``None`` and logs a warning if ``httpx`` is not
        installed.

        Args:
            timeout: Request timeout in seconds.
            headers: HTTP headers to attach to every request.

        Returns:
            Optional[httpx.Client]: ``None`` when the dependency
            is missing.
        """
        try:
            import httpx

            return httpx.Client(headers=headers or {}, timeout=timeout)
        except ImportError:
            logger.warning("%s: httpx not installed", type(self).__name__)
            return None

    def probe(self, path: str, *, timeout: float = 2.0) -> bool:
        """Issue ``GET base_url + path`` as a liveness probe.

        Args:
            path: URL path appended to ``base_url + '/'``.
            timeout: Override the default probe timeout.

        Returns:
            bool: True if the response status is 200.
        """
        if self.client is None:
            return False
        try:
            resp = self.client.get(f"{self.base_url}/{path.lstrip('/')}", timeout=timeout)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("%s probe failed: %s", type(self).__name__, exc)
            return False
