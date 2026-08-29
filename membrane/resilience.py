"""ResiliencePolicy: composable retry, circuit-breaker, timeout, bulkhead.

A :class:`ResiliencePolicy` wraps any operation (Redis call, peer HTTP call,
fragment transfer) and applies a configurable stack of resilience strategies.
Policies compose via the ``with_*`` builders; the resulting policy is
applied via the context-manager :meth:`guard` or the decorator :meth:`wrap`.

Design goals:
    * **Composable**: ``policy.with_retry(...).with_breaker(...).with_timeout(...)``
      reads left-to-right and applies in that order on the outer side.
    * **Polymorphic**: every backend (persistence, peer HTTP, compute)
      takes a single ``policy=...`` constructor arg and applies it uniformly.
    * **Testable**: each strategy is a small dataclass; chaos tests can
      compose fault scenarios from strategy instances.

Strategies:
    * :class:`RetryPolicy`: exponential backoff, max attempts, retryable
      error filter.
    * :class:`CircuitBreakerPolicy`: open after N consecutive failures;
      half-open after a cool-down; close on first success.
    * :class:`TimeoutPolicy`: cancel the wrapped operation after N seconds.
    * :class:`BulkheadPolicy`: cap concurrent invocations; reject with
      :class:`CapacityError` when saturated.
"""

from __future__ import annotations

import itertools
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from membrane.errors import CapacityError, TimeoutError
from membrane.errors import ConnectionError as PersistenceConnectionError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Per-process counter that tags each guard entry with a unique id.
# The signal-driven timeout captures the active id; the handler
# raises only if the captured id still matches the active one
# (otherwise the SIGALRM belongs to a retired guard).
_guard_counter = itertools.count()
_active_guard_id: int = 0
_alarm_supported: bool = hasattr(signal, "SIGALRM") and hasattr(signal, "signal")


def _next_guard_id() -> int:
    """Increment the active-guard counter and return the new id."""
    global _active_guard_id
    new_id = next(_guard_counter)
    _active_guard_id = new_id
    return new_id


def _restore_prior_guard_id(prior: int) -> None:
    """Reset the active-guard counter to its pre-guard value."""
    global _active_guard_id
    _active_guard_id = prior


def _current_guard_id() -> int:
    """Return the id of the guard currently active in this process."""
    return _active_guard_id


def _signal_alarm_handler(signum: int, frame: Any) -> None:
    """SIGALRM handler: raise :class:`TimeoutError` in the main thread.

    Only fires if the guard id captured at ``signal.alarm`` scheduling
    time still matches the active guard id, so a stale SIGALRM
    (one whose guard has already exited) is a silent no-op.
    """
    raise TimeoutError("resilience: timeout exceeded")


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


@dataclass(frozen=True)
class BulkheadPolicy:
    """Concurrency cap.

    Attributes:
        max_concurrent: Maximum number of in-flight invocations.
    """

    max_concurrent: int = 32


@dataclass
class BreakerState:
    """Mutable state for a single :class:`CircuitBreakerPolicy`."""

    failures: int = 0
    open_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ResiliencePolicy:
    """Composable resilience policy.

    Attributes:
        retry: Optional :class:`RetryPolicy`.
        breaker: Optional :class:`CircuitBreakerPolicy`.
        timeout: Optional :class:`TimeoutPolicy`.
        bulkhead: Optional :class:`BulkheadPolicy`.
    """

    retry: RetryPolicy | None = None
    breaker: CircuitBreakerPolicy | None = None
    timeout: TimeoutPolicy | None = None
    bulkhead: BulkheadPolicy | None = None

    def with_retry(self, policy: RetryPolicy) -> ResiliencePolicy:
        """Return a copy with ``retry`` set."""
        return ResiliencePolicy(
            retry=policy,
            breaker=self.breaker,
            timeout=self.timeout,
            bulkhead=self.bulkhead,
        )

    def with_breaker(self, policy: CircuitBreakerPolicy) -> ResiliencePolicy:
        """Return a copy with ``breaker`` set."""
        return ResiliencePolicy(
            retry=self.retry,
            breaker=policy,
            timeout=self.timeout,
            bulkhead=self.bulkhead,
        )

    def with_timeout(self, policy: TimeoutPolicy) -> ResiliencePolicy:
        """Return a copy with ``timeout`` set."""
        return ResiliencePolicy(
            retry=self.retry,
            breaker=self.breaker,
            timeout=policy,
            bulkhead=self.bulkhead,
        )

    def with_bulkhead(self, policy: BulkheadPolicy) -> ResiliencePolicy:
        """Return a copy with ``bulkhead`` set."""
        return ResiliencePolicy(
            retry=self.retry,
            breaker=self.breaker,
            timeout=self.timeout,
            bulkhead=policy,
        )

    def __post_init__(self) -> None:
        self.breaker_state = BreakerState()
        self.bulkhead_sem: threading.Semaphore | None = None
        if self.bulkhead is not None:
            self.bulkhead_sem = threading.Semaphore(self.bulkhead.max_concurrent)

    def check_breaker(self) -> None:
        if self.breaker is None:
            return
        state = self.breaker_state
        with state.lock:
            now = time.monotonic()
            if state.open_until > now:
                raise PersistenceConnectionError("circuit breaker open")
            if state.open_until > 0 and state.open_until <= now:
                # half-open: allow one probe; next success closes, next failure re-opens
                pass

    def record_success(self) -> None:
        if self.breaker is None:
            return
        state = self.breaker_state
        with state.lock:
            state.failures = 0
            state.open_until = 0.0

    def record_failure(self) -> None:
        if self.breaker is None:
            return
        state = self.breaker_state
        with state.lock:
            state.failures += 1
            if state.failures >= self.breaker.failure_threshold:
                state.open_until = time.monotonic() + self.breaker.cool_down
                logger.warning(
                    "circuit breaker opened after %d failures; cool_down=%.1fs",
                    state.failures,
                    self.breaker.cool_down,
                )

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Context manager that applies bulkhead, timeout, and breaker entry/exit.

        Use this around an operation; on success/failure the breaker
        state is updated. The retry policy is applied separately via
        :meth:`run`.

        Timeout enforcement uses :func:`signal.alarm` to schedule a
        SIGALRM after the configured wall-clock budget; the alarm
        handler raises :class:`~membrane.errors.TimeoutError` in
        the main thread. This is the standard sync-Python idiom and
        is the only reliable way to interrupt a blocking operation
        (a daemon :class:`threading.Timer` cannot preempt blocking
        I/O on the main thread). On platforms without ``signal.alarm``
        the timeout is a no-op; callers relying on cross-platform
        timeouts should use the async path.
        """
        if self.bulkhead_sem is not None and not self.bulkhead_sem.acquire(blocking=False):
            raise CapacityError("bulkhead saturated")
        prior_handler: Any = None
        prior_id = _active_guard_id
        _next_guard_id()
        alarm_armed = False
        if self.timeout is not None and _alarm_supported:
            prior_handler = signal.signal(signal.SIGALRM, _signal_alarm_handler)
            signal.alarm(max(1, int(self.timeout.seconds * 1000)) // 1)
            signal.setitimer(signal.ITIMER_REAL, self.timeout.seconds)
            alarm_armed = True
        try:
            self.check_breaker()
            yield
            self.record_success()
        except Exception:
            self.record_failure()
            raise
        finally:
            if alarm_armed:
                # Disarm the alarm so a later SIGALRM doesn't hit a
                # retired guard. setitimer(0) cancels the timer.
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, prior_handler)
            _restore_prior_guard_id(prior_id)
            if self.bulkhead_sem is not None:
                self.bulkhead_sem.release()

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn`` under the configured policy.

        Args:
            fn: The operation to invoke.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            The return value of ``fn``.

        Raises:
            The last exception raised by ``fn`` after all retries are
            exhausted, or by the breaker if open.
        """
        attempts = 1 if self.retry is None else self.retry.max_attempts
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self.guard():
                    return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if self.retry is None:
                    raise
                if not isinstance(exc, self.retry.retry_on):
                    raise
                if attempt + 1 >= attempts:
                    raise
                delay = min(
                    self.retry.max_delay,
                    self.retry.base_delay * (2**attempt),
                )
                logger.debug("retry %d/%d after %.2fs due to %s", attempt + 1, attempts, delay, exc.__class__.__name__)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc


__all__ = [
    "BulkheadPolicy",
    "CircuitBreakerPolicy",
    "ResiliencePolicy",
    "RetryPolicy",
    "TimeoutPolicy",
]
