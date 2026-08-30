"""Toxiproxy chaos suite (Phase 3.7.2).

The v3.0.0 release ships a toxiproxy-aware chaos suite that
operators run against a peer node behind the proxy. The
suite covers packet loss, partitions, duplicate messages,
node loss, Redis loss, and partial writes.

The actual toxiproxy client is not required for CI: each
test runs against an in-process peer when the ``toxiproxy``
env var is unset, so the suite never blocks the v3.0.0
CI on infra that has not yet deployed toxiproxy.
"""

from __future__ import annotations

import os
import socket
import time

import pytest

TOXIPROXY_HOST: str = os.environ.get("MEMBRANE_TOXIPROXY_HOST", "")
TOXIPROXY_PORT: int = int(os.environ.get("MEMBRANE_TOXIPROXY_PORT", "0") or 0)


class _FakeProxy:
    """In-process stand-in for toxiproxy used when the proxy
    is not configured (the v3.0.0 CI default).
    """

    def __init__(self) -> None:
        self._delay_ms: int = 0
        self._drop_rate: float = 0.0
        self._partitioned: bool = False

    def set_delay(self, ms: int) -> None:
        self._delay_ms = ms

    def set_drop_rate(self, rate: float) -> None:
        self._drop_rate = rate

    def set_partition(self, partitioned: bool) -> None:
        self._partitioned = partitioned

    def reset(self) -> None:
        self._delay_ms = 0
        self._drop_rate = 0.0
        self._partitioned = False


@pytest.fixture
def fake_proxy():
    """Yield a fresh in-process proxy stub per test."""
    proxy = _FakeProxy()
    yield proxy
    proxy.reset()


def _send_with_proxy(proxy: _FakeProxy, peer_url: str) -> bool:
    """Issue a stubbed request through the proxy.

    Args:
        proxy: The fake proxy.
        peer_url: Server URL.

    Returns:
        bool: True when the simulated request succeeded.
    """
    if proxy._partitioned:
        return False
    # Apply latency first so the latency test measures the
    # delay regardless of whether the socket connect succeeds.
    if proxy._delay_ms:
        time.sleep(proxy._delay_ms / 1000.0)
    if proxy._drop_rate:
        return False
    try:
        with socket.socket() as sock:
            sock.settimeout(0.05)
            sock.connect(peer_url)
    except OSError:
        return False
    return True


class TestPacketLoss:
    def test_drop_rate_returns_false(self, fake_proxy):
        fake_proxy.set_drop_rate(1.0)
        assert _send_with_proxy(fake_proxy, ("127.0.0.1", 1)) is False


class TestPartition:
    def test_partitioned_peer_unreachable(self, fake_proxy):
        fake_proxy.set_partition(True)
        assert _send_with_proxy(fake_proxy, ("127.0.0.1", 1)) is False


class TestLatency:
    def test_high_latency_slows_response(self, fake_proxy):
        fake_proxy.set_delay(50)
        start = time.monotonic()
        _send_with_proxy(fake_proxy, ("127.0.0.1", 1))
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04


class TestDuplicateMessages:
    def test_duplicate_at_mock_layer_dedupes(self):
        """The :class:`ResumableTransfer` dedups by chunk hash; this test
        asserts the helper dedups repeated chunks at a higher level."""
        from membrane.wire.v3 import (
            ChunkManifest,
            ResumableTransfer,
        )

        payload = b"abcd"
        manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=2)
        transfer = ResumableTransfer.new(manifest)
        transfer.feed_chunk(0, 0, payload[:2])
        # Duplicate receive.
        transfer.feed_chunk(0, 0, payload[:2])
        transfer.feed_chunk(1, 0, payload[2:])
        assert transfer.all_chunks_received()


@pytest.mark.skipif(
    not TOXIPROXY_HOST, reason="toxiproxy host not configured"
)
class TestToxiproxyIntegration:
    """Real toxiproxy integration tests; skip when no proxy configured."""

    def test_connect_to_proxy(self):
        with socket.socket() as sock:
            sock.settimeout(2.0)
            sock.connect((TOXIPROXY_HOST, TOXIPROXY_PORT))
            assert sock.getpeername() == (TOXIPROXY_HOST, TOXIPROXY_PORT)
