"""Resilience dataclasses for retry, circuit-breaker, and timeout.

The :mod:`membrane.wire.retry` module composes these into a real
client (Phase 3.3.7). The v3.0.0 release drops the old
:class:`ResiliencePolicy` composable and the
:class:`BulkheadPolicy` semaphore that the v2.0 arc carried as
placeholders; the v3 wire owns its own bulkhead as a
:class:`~membrane.wire.aio_client.WireBulkhead` and its own
timeout as a :class:`~membrane.wire.cancellation.CancellationToken`.

Each strategy here is a small frozen dataclass; the v3 wire
composes them with its own run loop so there is no shared
:class:`ResiliencePolicy` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

from membrane.errors import ConnectionError as PersistenceConnectionError


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry configuration.

    Attributes:
        max_attempts: Total attempts including the first; ``1`` disables retry.
        base_delay: Initial delay between attempts, in seconds.
        max_delay: Upper bound on backoff delay, in seconds.
        retry_on: Tuple of exception classes that trigger a retry. Other
            exceptions are propagated immediately.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 5.0
    retry_on: tuple[type[BaseException], ...] = (PersistenceConnectionError,)


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    """Circuit breaker configuration.

    Attributes:
        failure_threshold: Consecutive failures that trip the breaker open.
        cool_down: Seconds to wait before transitioning to half-open.
    """

    failure_threshold: int = 5
    cool_down: float = 30.0


@dataclass(frozen=True)
class TimeoutPolicy:
    """Timeout configuration.

    Attributes:
        seconds: Maximum allowed wall time for the wrapped operation.
    """

    seconds: float = 5.0


__all__ = [
    "CircuitBreakerPolicy",
    "RetryPolicy",
    "TimeoutPolicy",
]
