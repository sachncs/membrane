"""Tests for the wire_v3 async client + retry + circuit breaker (Phase 3.3.5-3.3.7)."""

from __future__ import annotations

import asyncio
import time

import pytest

from membrane.wire.v3.aio_client import (
    AsyncWireClient,
    CancellationToken,
    CircuitBreakerPolicy,
    RetryPolicy,
    WireBulkhead,
    compute_backoff,
    with_deadline,
)


class TestComputeBackoff:
    def test_returns_a_finite_float_within_max(self):
        policy = RetryPolicy(base_delay=0.1, max_delay=1.0, max_attempts=5)
        for attempt in range(5):
            delay = compute_backoff(policy, attempt)
            assert 0.0 <= delay <= 1.0


class TestCircuitBreakerPolicy:
    def test_closed_initially(self):
        cb = CircuitBreakerPolicy()
        assert cb.is_open() is False

    def test_opens_after_threshold(self):
        cb = CircuitBreakerPolicy(failure_threshold=3, cool_down=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is False
        cb.record_failure()
        assert cb.is_open() is True

    def test_success_resets(self):
        cb = CircuitBreakerPolicy(failure_threshold=2, cool_down=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.is_open() is False
        assert cb.failures == 0


class TestCancellationToken:
    def test_starts_uncancelled(self):
        token = CancellationToken()
        assert token.is_cancelled() is False

    def test_cancel_flips_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True


class TestWithDeadline:
    def test_runs_within_deadline(self):
        async def fast() -> int:
            return 42

        async def main():
            return await with_deadline(0.5, fast())

        assert asyncio.run(main()) == 42

    def test_deadline_raises(self):
        async def slow() -> int:
            await asyncio.sleep(1.0)
            return 42

        async def main():
            return await with_deadline(0.05, slow())

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(main())


class TestAsyncWireClient:
    def test_bulkhead_defaults(self):
        client = AsyncWireClient(base_url="http://node-1:8080")
        assert client.bulkhead.max_concurrent == 32

    def test_circuit_breaker_state_per_host(self):
        client = AsyncWireClient(base_url="http://node-1:8080")
        # The first request sets the breaker for the host.
        breaker = client.breaker.setdefault("http://node-1:8080", CircuitBreakerPolicy())
        assert breaker.is_open() is False

    def test_retry_policy_defaults(self):
        client = AsyncWireClient(base_url="http://node-1:8080")
        assert client.retry.max_attempts >= 1

    def test_circuit_breaker_records_failure_then_success(self):
        client = AsyncWireClient(base_url="http://node-1:8080")
        breaker = client.breaker.setdefault(
            "http://node-1:8080", CircuitBreakerPolicy(failure_threshold=2, cool_down=60.0)
        )
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open() is True
        breaker.record_success()
        assert breaker.is_open() is False
