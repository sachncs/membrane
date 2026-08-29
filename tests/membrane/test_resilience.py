"""Tests for the resilience policies wired through ResiliencePolicy.guard.

Covers TimeoutPolicy (newly wired), RetryPolicy, CircuitBreakerPolicy,
and BulkheadPolicy composition. None of the tests use real network
I/O; everything is exercised against small in-process sleeps so the
runtime stays deterministic.
"""

import threading
import time

import pytest

from membrane.errors import CapacityError, TimeoutError
from membrane.errors import ConnectionError as PersistenceConnectionError
from membrane.resilience import (
    BulkheadPolicy,
    CircuitBreakerPolicy,
    ResiliencePolicy,
    RetryPolicy,
    TimeoutPolicy,
)


def _sleep_for(seconds: float) -> None:
    time.sleep(seconds)


def test_no_policy_passes_through():
    """A bare ResiliencePolicy is a no-op wrapper."""
    policy = ResiliencePolicy()
    with policy.guard():
        x = 1 + 1
    assert x == 2


def test_timeout_policy_raises_when_guard_exceeds_budget():
    """TimeoutPolicy aborts the guard after the configured budget."""
    policy = ResiliencePolicy(timeout=TimeoutPolicy(seconds=0.05))
    start = time.monotonic()
    with pytest.raises(TimeoutError), policy.guard():
        _sleep_for(0.5)
    elapsed = time.monotonic() - start
    # Should fire well before the sleep would have completed.
    assert elapsed < 0.4


def test_timeout_policy_does_not_fire_when_guard_finishes_in_time():
    """A guard that completes quickly raises nothing."""
    policy = ResiliencePolicy(timeout=TimeoutPolicy(seconds=1.0))
    with policy.guard():
        _sleep_for(0.01)


def test_timeout_composes_with_bulkhead():
    """Bulkhead + timeout both apply; timeout wins when exceeded."""
    policy = ResiliencePolicy(
        timeout=TimeoutPolicy(seconds=0.05),
        bulkhead=BulkheadPolicy(max_concurrent=1),
    )
    # First call occupies the bulkhead slot but completes quickly;
    # verify it does not surface a TimeoutError.
    with policy.guard():
        _sleep_for(0.01)
    # Now a slow call exceeds the timeout before the bulkhead can
    # re-release; the timeout fires regardless.
    with pytest.raises(TimeoutError), policy.guard():
        _sleep_for(0.5)


def test_bulkhead_saturation_raises_capacity_error():
    """When the bulkhead is full, CapacityError fires immediately."""
    policy = ResiliencePolicy(bulkhead=BulkheadPolicy(max_concurrent=1))

    holder_in = threading.Event()
    main_can_enter = threading.Event()

    def hold_slot():
        with policy.guard():
            holder_in.set()
            main_can_enter.wait(timeout=2.0)

    t = threading.Thread(target=hold_slot, daemon=True)
    t.start()
    assert holder_in.wait(timeout=2.0)

    with pytest.raises(CapacityError), policy.guard():
        pass

    main_can_enter.set()
    t.join(timeout=2.0)


def test_retry_policy_retries_on_retryable_error():
    """RetryPolicy retries up to ``max_attempts`` on a retryable error."""
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PersistenceConnectionError("transient")
        return "ok"

    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=5, base_delay=0.001, max_delay=0.01),
    )
    result = policy.run(flaky)
    assert result == "ok"
    assert attempts["n"] == 3


def test_retry_policy_does_not_retry_non_retryable_error():
    """Errors outside ``retry_on`` propagate immediately."""
    attempts = {"n": 0}

    def boom():
        attempts["n"] += 1
        raise ValueError("permanent")

    policy = ResiliencePolicy(
        retry=RetryPolicy(max_attempts=5, base_delay=0.001, max_delay=0.01),
    )
    with pytest.raises(ValueError):
        policy.run(boom)
    assert attempts["n"] == 1


def test_circuit_breaker_opens_after_threshold():
    """After ``failure_threshold`` consecutive failures the breaker opens."""
    policy = ResiliencePolicy(
        breaker=CircuitBreakerPolicy(failure_threshold=3, cool_down=10.0),
        retry=RetryPolicy(max_attempts=1, base_delay=0.001, max_delay=0.01),
    )

    def boom():
        raise PersistenceConnectionError("nope")

    for _ in range(3):
        with pytest.raises(PersistenceConnectionError), policy.guard():
            raise boom()

    # Next call is short-circuited by the open breaker.
    with pytest.raises(PersistenceConnectionError, match="circuit breaker open"), policy.guard():
        pass


def test_retry_breaker_timeout_all_compose():
    """Timeout counts as a failure for the circuit breaker."""
    policy = ResiliencePolicy(
        timeout=TimeoutPolicy(seconds=0.02),
        breaker=CircuitBreakerPolicy(failure_threshold=2, cool_down=10.0),
        retry=RetryPolicy(max_attempts=1, base_delay=0.001, max_delay=0.01),
    )

    for _ in range(2):
        with pytest.raises(TimeoutError), policy.guard():
            _sleep_for(0.5)

    # Breaker is now open.
    with pytest.raises(PersistenceConnectionError, match="circuit breaker open"), policy.guard():
        pass
