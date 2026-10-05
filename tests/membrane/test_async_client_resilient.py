"""AsyncMembraneClient(resilient=True): retries 5xx and transport errors, fails fast when open."""

import asyncio

import httpx
import pytest

from membrane.client import AsyncMembraneClient, MembraneConnectionError
from membrane.compute.cpu import CPU
from membrane.content_store import InProcessBytes
from membrane.resilience import CircuitBreaker, CircuitBreakerPolicy, RetryPolicy


def test_retries_a_503_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, json={"digest": {}, "node_id": "n"})

    async def scenario() -> dict:
        client = AsyncMembraneClient(
            "http://node", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), resilient=True
        )
        client.wire.retry = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0)
        try:
            return await client.inventory()
        finally:
            await client.close()

    assert asyncio.run(scenario())["node_id"] == "n"
    assert calls["n"] == 2


def test_open_breaker_fails_fast() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    async def scenario() -> None:
        client = AsyncMembraneClient(
            "http://node", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), resilient=True
        )
        client.wire.retry = RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0)
        client.wire.breaker["http://node"] = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=2, cool_down=60))
        for _ in range(2):
            with pytest.raises(MembraneConnectionError):
                await client.inventory()
        with pytest.raises(MembraneConnectionError):
            await client.inventory()  # open: no network call
        await client.close()

    asyncio.run(scenario())
    assert calls["n"] == 2


def test_cpu_backend_bind_is_a_no_op() -> None:
    CPU().bind(InProcessBytes())
