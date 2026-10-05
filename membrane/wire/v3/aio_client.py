"""Async httpx + grpc.aio client with structured cancellation.

The v3.0.0 release replaces the v2.0 synchronous urllib-based
HTTP client with an async client built on
:class:`httpx.AsyncClient` and the v3 gRPC transport with
:class:`grpc.aio`. The :class:`WireBulkhead` and
:class:`CancellationToken` are first-class configs on the
client rather than a separate resilience policy.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from membrane.resilience import CircuitBreaker, RetryPolicy, compute_backoff

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CancellationToken:
    """Structured cancellation token.

    Attributes:
        cancelled: True when :meth:`cancel` has been called.
        event: asyncio.Event signalled when the token is cancelled.
    """

    cancelled: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def cancel(self) -> None:
        """Cancel the token and wake all awaiters."""
        if not self.cancelled:
            object.__setattr__(self, "cancelled", True)
            self.event.set()

    def is_cancelled(self) -> bool:
        """Return True when the token has been cancelled.

        Returns:
            bool: ``self.cancelled``.
        """
        return self.cancelled


@dataclass(frozen=True)
class WireBulkhead:
    """Concurrency cap for a v3 wire client.

    Attributes:
        max_concurrent: Maximum number of in-flight requests
            against this client (default 32).
        per_host: Per-host semaphore depth (default 8).
    """

    max_concurrent: int = 32
    per_host: int = 8


@dataclass
class AsyncWireClient:
    """Async httpx client with wire bulkhead + circuit breaker.

    Attributes:
        base_url: Server URL (e.g., ``http://node-1:8080``).
        bulkhead: Concurrency cap.
        retry: Retry policy for transient failures.
        timeout_sec: Per-request timeout.
        breaker: Per-host circuit breaker state.
    """

    base_url: str
    bulkhead: WireBulkhead = field(default_factory=WireBulkhead)
    retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(base_delay=0.1, max_delay=2.0))
    timeout_sec: float = 5.0
    breaker: dict[str, CircuitBreaker] = field(default_factory=dict)
    semaphore: asyncio.Semaphore | None = None
    client: Any = None

    async def ensure_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the bulkhead semaphore on first call.

        Returns:
            asyncio.Semaphore: The bulkhead semaphore.
        """
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.bulkhead.max_concurrent)
        return self.semaphore

    async def send(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        token: CancellationToken | None = None,
    ) -> Any:
        """Send one request through the bulkhead, breaker, and retry policy.

        Transport errors and 5xx responses are retried with jittered
        backoff and counted against the circuit breaker; while the breaker
        is open, calls fail fast. Connections are pooled across calls.

        Args:
            method: HTTP method.
            path: URL path relative to ``base_url``.
            headers: Request headers.
            params: Query parameters.
            json_body: JSON request body.
            token: Optional :class:`CancellationToken`.

        Returns:
            Any: The ``httpx.Response`` (the last one, when every attempt
            returned 5xx).

        Raises:
            RuntimeError: When the circuit breaker is open.
            httpx.HTTPError: When every attempt failed in transport.
            asyncio.CancelledError: When ``token`` is cancelled.
        """
        import httpx

        sem = await self.ensure_semaphore()
        breaker = self.breaker.setdefault(self.base_url, CircuitBreaker())
        if not breaker.allow():
            raise RuntimeError(f"circuit breaker open for {self.base_url}")
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout_sec)
        async with sem:
            url = f"{self.base_url}{path}"
            last_exc: Exception | None = None
            last_response: Any = None
            for attempt in range(self.retry.max_attempts):
                if token is not None and token.is_cancelled():
                    raise asyncio.CancelledError("cancelled before request")
                try:
                    response = await self.client.request(method, url, headers=headers, params=params, json=json_body)
                except httpx.HTTPError as exc:
                    breaker.record_failure()
                    last_exc = exc
                else:
                    if response.status_code < 500:
                        breaker.record_success()
                        return response
                    breaker.record_failure()
                    last_response = response
                if attempt + 1 < self.retry.max_attempts:
                    await asyncio.sleep(compute_backoff(self.retry, attempt))
            if last_response is not None:
                return last_response
            assert last_exc is not None
            raise last_exc

    async def request(
        self,
        method: str,
        path: str,
        token: CancellationToken | None = None,
    ) -> bytes:
        """Issue a body-less request and return the response body.

        Args:
            method: HTTP method.
            path: URL path relative to ``base_url``.
            token: Optional :class:`CancellationToken`.

        Returns:
            bytes: Response body.

        Raises:
            RuntimeError: When the breaker is open or every attempt got 5xx.
        """
        response = await self.send(method, path, token=token)
        if response.status_code >= 500:
            raise RuntimeError(f"server returned {response.status_code}")
        return bytes(response.content)

    async def close(self) -> None:
        """Close the pooled HTTP client."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None


async def with_deadline(duration_sec: float, awaitable: Any) -> Any:
    """Run ``awaitable`` with a wall-clock deadline.

    Args:
        duration_sec: Maximum wall-clock seconds.
        awaitable: The awaitable to time.

    Returns:
        Awaitable's result.

    Raises:
        asyncio.TimeoutError: When the deadline is exceeded.
    """
    return await asyncio.wait_for(awaitable, timeout=duration_sec)


__all__ = [
    "AsyncWireClient",
    "CancellationToken",
    "CircuitBreaker",
    "RetryPolicy",
    "WireBulkhead",
    "compute_backoff",
    "with_deadline",
]
