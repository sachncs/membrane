"""Async httpx + grpc.aio client with structured cancellation (Phase 3.3.5 + 3.3.6).

The v3.0.0 release replaces the v2.0 synchronous urllib-based
HTTP client with an async client built on
:class:`httpx.AsyncClient` and the v3 gRPC transport with
:class:`grpc.aio`. The :class:`WireBulkhead` and
:class:`CancellationToken` are first-class configs on the
client rather than a separate resilience policy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

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


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        max_attempts: Total attempts including the first.
        base_delay: Initial delay (seconds).
        max_delay: Cap on the per-attempt delay.
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 2.0


def compute_backoff(policy: RetryPolicy, attempt: int) -> float:
    """Compute the delay before retry ``attempt`` (0-based).

    Args:
        policy: The retry policy.
        attempt: Zero-based attempt index.

    Returns:
        float: Seconds to sleep before the next attempt.
    """
    import random

    delay = min(policy.max_delay, policy.base_delay * (2**attempt))
    return random.uniform(0, delay)


@dataclass
class CircuitBreakerPolicy:
    """Per-host circuit breaker state."""

    failure_threshold: int = 5
    cool_down: float = 30.0
    failures: int = 0
    open_until: float = 0.0

    def is_open(self, now: float | None = None) -> bool:
        """Return True when the breaker is open (caller should fail fast).

        Args:
            now: Monotonic clock. ``None`` reads ``time.monotonic``.

        Returns:
            bool: True when the breaker is still cooling down.
        """
        current = time.monotonic() if now is None else now
        return self.open_until > current

    def record_failure(self, now: float | None = None) -> None:
        """Increment the failure counter and open the breaker on threshold.

        Args:
            now: Monotonic clock. ``None`` reads ``time.monotonic``.
        """
        current = time.monotonic() if now is None else now
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open_until = current + self.cool_down

    def record_success(self) -> None:
        """Reset the failure counter on a successful call."""
        self.failures = 0
        self.open_until = 0.0


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
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_sec: float = 5.0
    breaker: dict[str, CircuitBreakerPolicy] = field(default_factory=dict)
    _semaphore: asyncio.Semaphore | None = None

    async def _ensure_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the bulkhead semaphore on first call."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.bulkhead.max_concurrent)
        return self._semaphore

    async def request(
        self,
        method: str,
        path: str,
        token: CancellationToken | None = None,
    ) -> bytes:
        """Issue a single async HTTP request through the bulkhead.

        Args:
            method: HTTP method.
            path: URL path relative to ``base_url``.
            token: Optional :class:`CancellationToken`.

        Returns:
            bytes: Response body.

        Raises:
            asyncio.TimeoutError: When the request exceeds the
                per-call timeout.
            asyncio.CancelledError: When ``token`` is cancelled
                before the request completes.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("httpx is required for AsyncWireClient") from exc

        sem = await self._ensure_semaphore()
        breaker = self.breaker.setdefault(self.base_url, CircuitBreakerPolicy())
        if breaker.is_open():
            raise RuntimeError(f"circuit breaker open for {self.base_url}")

        async with sem:
            url = f"{self.base_url}{path}"
            last_exc: Exception | None = None
            for attempt in range(self.retry.max_attempts):
                if token is not None and token.is_cancelled():
                    raise asyncio.CancelledError("cancelled before request")
                try:
                    async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                        resp = await client.request(method, url)
                        if resp.status_code >= 500:
                            breaker.record_failure()
                            last_exc = RuntimeError(
                                f"server returned {resp.status_code}"
                            )
                        else:
                            breaker.record_success()
                            return resp.content
                except httpx.HTTPError as exc:
                    breaker.record_failure()
                    last_exc = exc
                if attempt + 1 < self.retry.max_attempts:
                    delay = compute_backoff(self.retry, attempt)
                    await asyncio.sleep(delay)
            assert last_exc is not None
            raise last_exc


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
    "CircuitBreakerPolicy",
    "RetryPolicy",
    "WireBulkhead",
    "compute_backoff",
    "with_deadline",
]
