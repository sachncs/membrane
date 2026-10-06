"""Backpressure and rate limiting for the HTTP transport.

* :class:`ConcurrencyLimitMiddleware` bounds in-flight requests. A request
  that cannot get a slot within ``queue_timeout_sec`` gets ``503`` +
  ``Retry-After`` instead of queueing without bound behind a saturated
  node.
* :class:`RateLimitMiddleware` applies a token bucket per credential (the
  bearer token's digest, else the client address) and answers ``429`` +
  ``Retry-After`` when a caller exceeds it.

The ``/livez`` and ``/readyz`` probes bypass both limits, so an
overloaded node keeps answering health checks.
"""

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import anyio.to_thread

from membrane.auth.apikey import hash_key
from membrane.transport.request_id import ASGIApp, Receive, Scope, Send

EXEMPT_PATHS = frozenset({"/livez", "/readyz"})
MAX_TRACKED_CALLERS = 10_000

type RejectHook = Callable[[str], None]


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportLimits:
    """Capacity settings for the HTTP listener.

    Attributes:
        max_concurrency: Requests handled at once; ``0`` disables the
            bound. Also sizes the worker thread pool that runs handlers.
        queue_timeout_sec: Seconds a request waits for a slot before 503.
        rate_limit_per_sec: Sustained requests per second per credential;
            ``0`` disables rate limiting.
        rate_limit_burst: Bucket size; ``0`` means twice the rate.
        max_connections: Open connections uvicorn accepts before answering
            503 itself; ``None`` for no limit.
        keep_alive_timeout_sec: Idle keep-alive connection timeout.
        enable_api_docs: Serve ``/openapi.json`` (behind the ``read``
            scope). Off by default: the schema maps the attack surface.
    """

    max_concurrency: int = 64
    queue_timeout_sec: float = 0.1
    rate_limit_per_sec: float = 0.0
    rate_limit_burst: int = 0
    max_connections: int | None = None
    keep_alive_timeout_sec: float = 5.0
    enable_api_docs: bool = False


async def send_json(send: Send, status: int, body: dict[str, str], retry_after: int) -> None:
    """Send a complete JSON response with a ``Retry-After`` header.

    Args:
        send: ASGI send callable.
        status: HTTP status code.
        body: JSON body.
        retry_after: Seconds the client should wait.
    """
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"retry-after", str(retry_after).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def is_exempt(scope: Scope) -> bool:
    """Whether a scope bypasses limits (non-HTTP, or a health probe).

    Args:
        scope: ASGI scope.

    Returns:
        bool: True for lifespan/websocket scopes and probe paths.
    """
    return scope["type"] != "http" or scope.get("path") in EXEMPT_PATHS


class InFlight:
    """Requests being handled and requests waiting for a slot.

    Read by the routing scheduler as the node's queue depth. Updated
    only from the event loop thread, so plain integers suffice.

    Attributes:
        active: Requests holding a concurrency slot.
        waiting: Requests queued for a slot.
    """

    def __init__(self) -> None:
        """Start at zero."""
        self.active = 0
        self.waiting = 0


class ConcurrencyLimitMiddleware:
    """Bound in-flight requests; shed load with 503 when saturated."""

    def __init__(
        self,
        app: ASGIApp,
        max_concurrency: int,
        queue_timeout_sec: float,
        on_reject: RejectHook | None = None,
        in_flight: InFlight | None = None,
    ) -> None:
        """Wrap ``app``.

        Args:
            app: The downstream ASGI application.
            max_concurrency: Requests handled at once.
            queue_timeout_sec: Seconds to wait for a slot.
            on_reject: Called with ``"overloaded"`` for each shed request.
            in_flight: Counters updated as requests wait and run.
        """
        self.app = app
        self.max_concurrency = max_concurrency
        self.queue_timeout_sec = queue_timeout_sec
        self.in_flight = in_flight or InFlight()
        self.__on_reject = on_reject
        self.__slots = asyncio.Semaphore(max_concurrency)
        self.__pool_sized = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection scope.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if is_exempt(scope):
            await self.app(scope, receive, send)
            return
        if not self.__pool_sized:
            # Sync handlers run on AnyIO's worker threads (40 by default);
            # size the pool so admitted requests are not queued again there.
            limiter = anyio.to_thread.current_default_thread_limiter()
            limiter.total_tokens = max(limiter.total_tokens, self.max_concurrency)
            self.__pool_sized = True
        self.in_flight.waiting += 1
        try:
            await asyncio.wait_for(self.__slots.acquire(), timeout=self.queue_timeout_sec)
        except TimeoutError:
            if self.__on_reject is not None:
                self.__on_reject("overloaded")
            await send_json(send, 503, {"error": "overloaded"}, retry_after=1)
            return
        finally:
            self.in_flight.waiting -= 1
        self.in_flight.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.in_flight.active -= 1
            self.__slots.release()


def caller_key(scope: Scope) -> str:
    """Identify the caller a rate-limit bucket belongs to.

    Args:
        scope: ASGI scope.

    Returns:
        str: A digest of the Authorization header when present, else the
        client address.
    """
    for name, value in scope.get("headers") or []:
        if name.lower() == b"authorization":
            return "auth:" + hash_key(value.decode("latin-1"))
    client = scope.get("client")
    return f"addr:{client[0] if client else 'unknown'}"


class RateLimitMiddleware:
    """Token-bucket rate limit per credential; answers 429 when exceeded."""

    def __init__(self, app: ASGIApp, rate_per_sec: float, burst: int, on_reject: RejectHook | None = None) -> None:
        """Wrap ``app``.

        Args:
            app: The downstream ASGI application.
            rate_per_sec: Tokens added per second.
            burst: Bucket capacity.
            on_reject: Called with ``"rate_limited"`` for each rejected request.
        """
        self.app = app
        self.rate_per_sec = rate_per_sec
        self.burst = float(burst)
        self.__on_reject = on_reject
        self.__buckets: dict[str, tuple[float, float]] = {}

    def take(self, key: str, now: float) -> float:
        """Take one token from ``key``'s bucket.

        Args:
            key: Caller key.
            now: Monotonic time in seconds.

        Returns:
            float: ``0`` when allowed; otherwise seconds until a token is
            available.
        """
        tokens, last = self.__buckets.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate_per_sec)
        if tokens >= 1.0:
            self.__buckets[key] = (tokens - 1.0, now)
            wait = 0.0
        else:
            self.__buckets[key] = (tokens, now)
            wait = (1.0 - tokens) / self.rate_per_sec
        if len(self.__buckets) > MAX_TRACKED_CALLERS:
            self.__forget_idle(now)
        return wait

    def __forget_idle(self, now: float) -> None:
        """Drop buckets that have refilled completely (their callers went idle).

        Args:
            now: Monotonic time in seconds.
        """
        refill_sec = self.burst / self.rate_per_sec
        self.__buckets = {k: v for k, v in self.__buckets.items() if now - v[1] < refill_sec}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection scope.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if is_exempt(scope):
            await self.app(scope, receive, send)
            return
        wait = self.take(caller_key(scope), time.monotonic())
        if wait > 0:
            if self.__on_reject is not None:
                self.__on_reject("rate_limited")
            await send_json(send, 429, {"error": "rate limited"}, retry_after=max(1, math.ceil(wait)))
            return
        await self.app(scope, receive, send)


__all__ = [
    "EXEMPT_PATHS",
    "MAX_TRACKED_CALLERS",
    "ConcurrencyLimitMiddleware",
    "InFlight",
    "RateLimitMiddleware",
    "RejectHook",
    "TransportLimits",
    "caller_key",
    "is_exempt",
    "send_json",
]
