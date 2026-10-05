"""Peer calls fail fast while a peer's circuit breaker is open, and recover after."""

from membrane.errors import NetworkError
from membrane.network.peer import Peer
from membrane.resilience import CircuitBreaker, CircuitBreakerPolicy


class FlakyTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.up = False

    def request(self, method, url, body, headers, timeout_sec):
        self.calls += 1
        if not self.up:
            raise NetworkError("down")
        return {"ok": True}

    def request_bytes(self, method, url, body, headers, timeout_sec):
        raise NotImplementedError


def test_open_breaker_skips_the_network_then_half_open_trial_recovers() -> None:
    transport = FlakyTransport()
    peer = Peer(
        "http://p:1",
        transport=transport,
        max_retries=2,
        retry_delay_sec=0.0,
        breaker_policy=CircuitBreakerPolicy(failure_threshold=2, cool_down=0.2),
    )
    assert peer.heartbeat() is None and peer.heartbeat() is None
    assert transport.calls == 4  # two calls, two attempts each
    assert peer.breaker.state == "open"
    assert peer.heartbeat() is None
    assert transport.calls == 4  # failed fast: no network
    import time

    time.sleep(0.25)
    transport.up = True
    assert peer.heartbeat() == {"ok": True}  # the half-open trial
    assert peer.breaker.state == "closed"


def test_half_open_admits_one_trial_at_a_time() -> None:
    breaker = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=1, cool_down=0.0))
    breaker.record_failure(now=0.0)
    assert breaker.allow(now=1.0) is True
    assert breaker.allow(now=1.0) is False  # trial already in flight
    breaker.record_failure(now=1.0)
    assert breaker.allow(now=1.0) is True  # cool-down 0: next trial
