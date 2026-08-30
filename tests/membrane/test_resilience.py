"""Tests for the resilience dataclasses (Phase 3.1.1).

The v3.0.0 release narrows :mod:`membrane.resilience` to the
three standalone dataclasses (:class:`RetryPolicy`,
:class:`CircuitBreakerPolicy`, :class:`TimeoutPolicy`); the
:class:`ResiliencePolicy` composable and the
:class:`BulkheadPolicy` semaphore are gone. The real run loop
lives in :mod:`membrane.wire.retry` (Phase 3.3.7).
"""

import pytest

from membrane.resilience import (
    CircuitBreakerPolicy,
    RetryPolicy,
    TimeoutPolicy,
)


def test_retry_policy_defaults():
    """Defaults: 3 attempts, 0.5s base, 5.0s max, retry on PersistenceConnectionError."""
    p = RetryPolicy()
    assert p.max_attempts == 3
    assert p.base_delay == 0.5
    assert p.max_delay == 5.0
    assert p.max_attempts >= 1


def test_retry_policy_custom_attempts():
    p = RetryPolicy(max_attempts=5, base_delay=0.1, max_delay=2.0)
    assert p.max_attempts == 5
    assert p.base_delay == 0.1
    assert p.max_delay == 2.0


def test_circuit_breaker_policy_defaults():
    p = CircuitBreakerPolicy()
    assert p.failure_threshold == 5
    assert p.cool_down == 30.0


def test_circuit_breaker_policy_custom_threshold():
    p = CircuitBreakerPolicy(failure_threshold=3, cool_down=10.0)
    assert p.failure_threshold == 3
    assert p.cool_down == 10.0


def test_timeout_policy_defaults():
    p = TimeoutPolicy()
    assert p.seconds == 5.0


def test_timeout_policy_custom():
    p = TimeoutPolicy(seconds=0.05)
    assert p.seconds == 0.05


def test_all_policies_are_frozen():
    """Every policy is a frozen dataclass: cannot mutate fields after construction."""
    for policy in (RetryPolicy(), CircuitBreakerPolicy(), TimeoutPolicy()):
        with pytest.raises((AttributeError, TypeError)):
            policy.__setattr__("seconds", 99.0)  # type: ignore[attr-defined]
