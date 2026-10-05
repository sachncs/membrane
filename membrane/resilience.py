"""Retry backoff and circuit breaking for calls to other nodes.

* :class:`RetryPolicy`: how many attempts and how long to back off
  (exponential with full jitter, :func:`compute_backoff`).
* :class:`CircuitBreakerPolicy`: when to stop calling a failing peer.
* :class:`CircuitBreaker`: the per-peer state machine. After
  ``failure_threshold`` consecutive failures it opens and calls fail fast;
  after ``cool_down`` seconds one trial call is let through (half-open),
  and its outcome closes or re-opens the breaker.

:class:`~membrane.network.peer.Peer` and
:class:`~membrane.wire.v3.aio_client.AsyncWireClient` use them.
"""

import random
import threading
import time
from dataclasses import dataclass, field

from membrane.errors import ConnectionError as PersistenceConnectionError


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry configuration.

    Attributes:
        max_attempts: Total attempts including the first; ``1`` disables retry.
        base_delay: Backoff ceiling for the first retry, in seconds.
        max_delay: Upper bound on the backoff ceiling, in seconds.
        retry_on: Exception classes that trigger a retry (for callers that
            retry on exceptions).
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 5.0
    retry_on: tuple[type[BaseException], ...] = (PersistenceConnectionError,)


def compute_backoff(policy: RetryPolicy, attempt: int) -> float:
    """Return the delay before retry ``attempt`` (0-based), with full jitter.

    Args:
        policy: The retry policy.
        attempt: Zero-based retry index.

    Returns:
        float: Seconds to sleep, uniform in ``[0, min(max, base * 2**attempt)]``.
    """
    return random.uniform(0, min(policy.max_delay, policy.base_delay * (2**attempt)))


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    """Circuit breaker configuration.

    Attributes:
        failure_threshold: Consecutive failures that trip the breaker open.
        cool_down: Seconds to wait before letting a trial call through.
    """

    failure_threshold: int = 5
    cool_down: float = 30.0


@dataclass
class CircuitBreaker:
    """Thread-safe circuit breaker for one peer.

    Attributes:
        policy: Thresholds.
        failures: Consecutive failures so far.
        open_until: Monotonic time until which calls fail fast.
    """

    policy: CircuitBreakerPolicy = field(default_factory=CircuitBreakerPolicy)
    failures: int = 0
    open_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    trial_in_flight: bool = field(default=False, repr=False, compare=False)

    @property
    def state(self) -> str:
        """``"closed"``, ``"open"``, or ``"half_open"``."""
        with self.lock:
            if self.failures < self.policy.failure_threshold:
                return "closed"
            return "open" if time.monotonic() < self.open_until else "half_open"

    def is_open(self, now: float | None = None) -> bool:
        """Whether calls should fail fast right now.

        Args:
            now: Monotonic time; :func:`time.monotonic` when ``None``.

        Returns:
            bool: True while cooling down.
        """
        current = time.monotonic() if now is None else now
        with self.lock:
            return self.open_until > current

    def allow(self, now: float | None = None) -> bool:
        """Whether a call may proceed; claims the single half-open trial.

        Args:
            now: Monotonic time; :func:`time.monotonic` when ``None``.

        Returns:
            bool: True when closed, or for the one trial call after the
            cool-down; False while open or while the trial is in flight.
        """
        current = time.monotonic() if now is None else now
        with self.lock:
            if self.failures < self.policy.failure_threshold:
                return True
            if current < self.open_until or self.trial_in_flight:
                return False
            self.trial_in_flight = True
            return True

    def record_failure(self, now: float | None = None) -> None:
        """Count a failure; open (or re-open) the breaker at the threshold.

        Args:
            now: Monotonic time; :func:`time.monotonic` when ``None``.
        """
        current = time.monotonic() if now is None else now
        with self.lock:
            self.failures += 1
            self.trial_in_flight = False
            if self.failures >= self.policy.failure_threshold:
                self.open_until = current + self.policy.cool_down

    def record_success(self) -> None:
        """Close the breaker."""
        with self.lock:
            self.failures = 0
            self.open_until = 0.0
            self.trial_in_flight = False


@dataclass(frozen=True)
class TimeoutPolicy:
    """Timeout configuration.

    Attributes:
        seconds: Maximum allowed wall time for the wrapped operation.
    """

    seconds: float = 5.0


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "RetryPolicy",
    "TimeoutPolicy",
    "compute_backoff",
]
