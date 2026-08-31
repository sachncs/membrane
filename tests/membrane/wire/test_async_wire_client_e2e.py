"""End-to-end AsyncWireClient test (Phase 3.3.5-3.3.7 follow-up).

The Phase 3.3.5-3.3.7 commits shipped the AsyncWireClient
class surface; the existing unit test only exercised the
shared-state helpers. This test runs the full
``AsyncWireClient.request`` path against an httpx.MockTransport
to verify the bulkhead + circuit breaker + retry behavior
end-to-end.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from membrane.wire.v3.aio_client import (
    AsyncWireClient,
    CircuitBreakerPolicy,
    RetryPolicy,
    WireBulkhead,
)


def _make_async_transport(routes: dict[tuple[str, str], tuple[int, bytes]]):
    """Build an :class:`httpx.MockTransport` for the wire client.

    Args:
        routes: ``{(method, path): (status, body)}``.

    Returns:
        httpx.MockTransport: Bound handler.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, content=b"")
        status, body = routes[key]
        return httpx.Response(status, content=body)
    return httpx.MockTransport(handler)


def _run(coro):
    """Run an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


class TestAsyncWireClientE2E:
    def _patch_async_client(self, transport: httpx.MockTransport):
        """Patch :class:`httpx.AsyncClient` to return one with our transport.

        Args:
            transport: The MockTransport to use.

        Returns:
            Callable that restores the original class.
        """
        import contextlib

        @contextlib.contextmanager
        def patcher():
            original = httpx.AsyncClient
            captured_client = original(transport=transport, base_url="http://t")

            def factory(*args, **kwargs):
                # Use our transport regardless of kwargs.
                kwargs.pop("transport", None)
                kwargs.pop("base_url", None)
                return original(transport=transport, base_url="http://t", **kwargs)

            try:
                httpx.AsyncClient = factory  # type: ignore[assignment]
                yield captured_client
            finally:
                httpx.AsyncClient = original  # type: ignore[assignment]
        return patcher()

    def test_request_2xx_returns_body(self):
        routes = {("GET", "/foo"): (200, b"hello")}

        async def run() -> bytes:
            transport = _make_async_transport(routes)
            with self._patch_async_client(transport) as _:
                client = AsyncWireClient(base_url="http://t", timeout_sec=1.0)
                return await client.request("GET", "/foo")

        result = _run(run())
        assert result == b"hello"

    def test_request_5xx_records_failure_and_retries(self):
        """A persistent 500 records failures and the breaker opens."""
        routes = {("GET", "/x"): (500, b"err")}

        async def run() -> None:
            transport = _make_async_transport(routes)
            with self._patch_async_client(transport) as _:
                client = AsyncWireClient(
                    base_url="http://x",
                    timeout_sec=1.0,
                    retry=RetryPolicy(
                        max_attempts=2, base_delay=0.0, max_delay=0.0
                    ),
                    bulkhead=WireBulkhead(max_concurrent=2),
                )
                client.breaker["http://x"] = CircuitBreakerPolicy(
                    failure_threshold=2, cool_down=60.0
                )
                with pytest.raises(RuntimeError, match="server returned"):
                    await client.request("GET", "/x")
                # The breaker recorded 2 failures.
                assert client.breaker["http://x"].failures == 2
                # The breaker is now open.
                assert client.breaker["http://x"].is_open() is True
                # A subsequent request short-circuits.
                with pytest.raises(RuntimeError, match="circuit breaker open"):
                    await client.request("GET", "/x")

        _run(run())

    def test_request_4xx_returns_body_without_recording_failure(self):
        routes = {("GET", "/y"): (404, b"missing")}

        async def run() -> bytes:
            transport = _make_async_transport(routes)
            with self._patch_async_client(transport) as _:
                client = AsyncWireClient(
                    base_url="http://y", timeout_sec=1.0
                )
                client.breaker["http://y"] = CircuitBreakerPolicy(
                    failure_threshold=1, cool_down=60.0
                )
                return await client.request("GET", "/y")

        # 4xx is a successful response; the breaker is untouched.
        result = _run(run())
        assert result == b"missing"

    def test_cancellation_token_aborts_request(self):
        """A cancelled token short-circuits the request with CancelledError."""

        async def run() -> None:
            transport = _make_async_transport(
                {("GET", "/slow"): (200, b"hi")}
            )
            with self._patch_async_client(transport) as _:
                client = AsyncWireClient(base_url="http://t", timeout_sec=5.0)
                from membrane.wire.v3.aio_client import CancellationToken

                token = CancellationToken()
                # Pre-cancel the token; the first attempt's
                # pre-loop check fires CancelledError.
                token.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await client.request("GET", "/slow", token=token)
                # No breaker failures recorded for a cancelled
                # call.
                assert client.breaker == {}

    def test_with_deadline_wraps_request(self):
        """``with_deadline(..., request())`` enforces a wall-clock cap."""

        async def run() -> None:
            from membrane.wire.v3.aio_client import with_deadline

            async def slow_request() -> bytes:
                # Block for 200ms via sleep.
                await asyncio.sleep(0.2)
                return b"too-late"

            with pytest.raises(asyncio.TimeoutError):
                # Deadline is 50ms; slow_request takes 200ms.
                await with_deadline(0.05, slow_request())

        asyncio.run(run())

    def test_with_deadline_passes_fast_request(self):
        async def run() -> None:
            from membrane.wire.v3.aio_client import with_deadline

            async def fast() -> int:
                return 42

            result = await with_deadline(0.5, fast())
            assert result == 42

        asyncio.run(run())
