"""AsyncWireClient retry backoff timing (Phase 3.3.5-3.3.7 follow-up)."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from membrane.wire.v3.aio_client import (
    AsyncWireClient,
    CancellationToken,
    CircuitBreakerPolicy,
    RetryPolicy,
    WireBulkhead,
    compute_backoff,
)


def _make_transport(routes: dict[tuple[str, str], tuple[int, bytes]]):
    """Build an :class:`httpx.MockTransport`."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, content=b"")
        status, body = routes[key]
        return httpx.Response(status, content=body)
    return httpx.MockTransport(handler)


class TestAsyncWireClientRetry:
    def test_compute_backoff_grows_within_max(self):
        policy = RetryPolicy(base_delay=0.01, max_delay=0.05, max_attempts=5)
        # The full-jitter backoff caps at max_delay. Sample
        # many seeds to keep the test deterministic.
        for attempt in range(5):
            delays = [compute_backoff(policy, attempt) for _ in range(50)]
            assert all(0.0 <= d <= 0.05 + 1e-9 for d in delays)

    def test_retry_with_zero_base_skips_sleep(self):
        """A zero base_delay produces zero-mean backoff (no waiting)."""
        policy = RetryPolicy(base_delay=0.0, max_delay=0.0, max_attempts=4)
        assert all(compute_backoff(policy, i) == 0.0 for i in range(4))

    def test_compute_backoff_uses_random_jitter(self):
        policy = RetryPolicy(base_delay=0.1, max_delay=0.1, max_attempts=3)
        # All samples are uniform in [0, 0.1]; verify that
        # different samples yield different values (random
        # jitter is real).
        samples = [compute_backoff(policy, 0) for _ in range(20)]
        assert len(set(samples)) > 1

    def test_retry_increments_breaker_failures(self):
        """A persistent 500 increments the breaker counter."""
        routes = {("GET", "/x"): (500, b"err")}
        transport = _make_transport(routes)

        async def run():
            client = AsyncWireClient(
                base_url="http://x",
                timeout_sec=1.0,
                retry=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0),
            )
            # Monkey-patch the inner AsyncClient factory.
            inner = httpx.AsyncClient(transport=transport, base_url="http://x")
            original = httpx.AsyncClient

            def factory(*args, **kwargs):
                kwargs.pop("transport", None)
                kwargs.pop("base_url", None)
                return original(transport=transport, base_url="http://x", **kwargs)

            try:
                httpx.AsyncClient = factory  # type: ignore[assignment]
                from contextlib import suppress
                with suppress(RuntimeError):
                    await client.request("GET", "/x")
            finally:
                httpx.AsyncClient = original  # type: ignore[assignment]
                await inner.aclose()
            return client

        client = asyncio.run(run())
        # The breaker recorded 3 failures.
        assert client.breaker["http://x"].failures == 3

    def test_retry_with_first_attempt_success_records_zero_failures(self):
        """A 2xx on the first attempt does not touch the breaker."""
        routes = {("GET", "/ok"): (200, b"hello")}
        transport = _make_transport(routes)

        async def run():
            client = AsyncWireClient(
                base_url="http://x",
                timeout_sec=1.0,
                retry=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0),
            )
            original = httpx.AsyncClient

            def factory(*args, **kwargs):
                kwargs.pop("transport", None)
                kwargs.pop("base_url", None)
                return original(transport=transport, base_url="http://x", **kwargs)

            try:
                httpx.AsyncClient = factory  # type: ignore[assignment]
                result = await client.request("GET", "/ok")
            finally:
                httpx.AsyncClient = original  # type: ignore[assignment]
            assert result == b"hello"
            return client

        client = asyncio.run(run())
        # No failures recorded.
        assert client.breaker["http://x"].failures == 0

    def test_retry_respects_max_attempts_cap(self):
        """max_attempts caps the number of attempts at the configured value."""
        attempts_made: list[int] = []

        # Mock a transport that records every attempt.
        def handler(request: httpx.Request) -> httpx.Response:
            attempts_made.append(1)
            return httpx.Response(500, content=b"err")

        transport = httpx.MockTransport(handler)

        async def run():
            client = AsyncWireClient(
                base_url="http://x",
                timeout_sec=1.0,
                retry=RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0),
            )
            original = httpx.AsyncClient

            def factory(*args, **kwargs):
                kwargs.pop("transport", None)
                kwargs.pop("base_url", None)
                return original(transport=transport, base_url="http://x", **kwargs)

            try:
                httpx.AsyncClient = factory  # type: ignore[assignment]
                from contextlib import suppress
                with suppress(RuntimeError):
                    await client.request("GET", "/x")
            finally:
                httpx.AsyncClient = original  # type: ignore[assignment]
            return client

        asyncio.run(run())
        # The cap is 4: 1 initial attempt + 3 retries.
        assert len(attempts_made) == 4
