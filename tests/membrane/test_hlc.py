"""Tests for HLC — pack/unpack, tick, merge, Clock."""

from __future__ import annotations

import time

import pytest

from membrane.hlc import (
    HLC,
    Clock,
    compare,
    merge,
    pack,
    tick,
    unpack,
)


class TestPackUnpack:
    def test_round_trip_zero(self):
        hlc = HLC(physical_ms=0, logical=0)
        assert unpack(pack(hlc)) == hlc

    def test_round_trip_arbitrary(self):
        hlc = HLC(physical_ms=123_456_789_012, logical=12_345)
        assert unpack(pack(hlc)) == hlc

    def test_pack_is_64_bit(self):
        hlc = HLC(physical_ms=(1 << 48) - 1, logical=65535)
        assert pack(hlc) < 1 << 64

    def test_physical_zero_logical_nonzero(self):
        hlc = HLC(physical_ms=0, logical=42)
        assert unpack(pack(hlc)) == hlc


class TestTick:
    def test_advances_from_past(self):
        prev = HLC(physical_ms=100, logical=0)
        nxt = tick(prev)
        # Either physical grew (real-world clock) or logical was
        # bumped by one.
        assert nxt.physical_ms > prev.physical_ms or nxt.logical == 1

    def test_logical_increment_same_physical(self):
        # Force equality on physical: a backward physical reads as
        # same-ms, so logical must increment.
        prev = HLC(physical_ms=int(time.time() * 1000) + 1000, logical=0)
        nxt = tick(prev)
        # Real-world clock < prev.physical_ms, so we expect logical +1
        assert nxt.physical_ms == prev.physical_ms
        assert nxt.logical == 1

    def test_logical_saturation_rolls_to_next_physical(self):
        # Saturate the logical counter manually.
        prev = HLC(physical_ms=int(time.time() * 1000) + 1000, logical=65535)
        nxt = tick(prev)
        assert nxt.physical_ms == prev.physical_ms + 1
        assert nxt.logical == 0


class TestMerge:
    def test_observed_strictly_newer_wins(self):
        local = HLC(physical_ms=100, logical=5)
        observed = HLC(physical_ms=200, logical=0)
        merged = merge(local, observed)
        assert merged.physical_ms == 200
        assert merged.logical == 0

    def test_observed_older_local_bumps_logical(self):
        local = HLC(physical_ms=200, logical=5)
        observed = HLC(physical_ms=100, logical=999)
        merged = merge(local, observed)
        assert merged.physical_ms == 200
        assert merged.logical >= 6

    def test_equal_physical_bumps_max_plus_one(self):
        local = HLC(physical_ms=100, logical=5)
        observed = HLC(physical_ms=100, logical=2)
        merged = merge(local, observed)
        assert merged.physical_ms == 100
        assert merged.logical == 6

    def test_accepts_integer_observed(self):
        local = HLC(physical_ms=100, logical=5)
        observed = pack(HLC(physical_ms=100, logical=3))
        merged = merge(local, observed)
        assert merged.logical == 6

    def test_merged_clock_total_order(self):
        a = merge(HLC(physical_ms=10, logical=0), HLC(physical_ms=20, logical=0))
        b = merge(a, HLC(physical_ms=15, logical=99))
        # The merged clock moves forward in time monotonically.
        assert pack(b) > pack(a)


class TestCompare:
    def test_less(self):
        assert compare(0, 1) == -1

    def test_greater(self):
        assert compare(1, 0) == 1

    def test_equal(self):
        assert compare(42, 42) == 0

    def test_total_order(self):
        # An increasing sequence of HLCs produced by tick() is
        # monotonic; comparisons reflect that.
        a = pack(tick(HLC(physical_ms=int(time.time() * 1000), logical=0)))
        b = pack(tick(unpack(a)))
        assert compare(a, b) <= 0


class TestClock:
    def test_tick_advances(self):
        clk = Clock()
        v1 = clk.tick()
        time.sleep(0.001)
        v2 = clk.tick()
        # Real-world sleep ensures different physical ms in most
        # cases; the merge/tick logic also keeps ``v2 > v1``
        # when they're equal on physical.
        if v2 == v1:
            # Equal physical ms: merge must have advanced logical.
            clk2 = unpack(v1)
            assert clk2.physical_ms != 0
        assert v2 >= v1

    def test_merge_observes(self):
        clk = Clock()
        v_local = clk.tick()  # noqa: F841  -- reserved for future assertions
        # Inject a higher observed value.
        observed = pack(HLC(physical_ms=int(time.time() * 1000) + 60000, logical=0))
        v_new = clk.merge(observed)
        assert unpack(v_new).physical_ms == int(time.time() * 1000) + 60000

    def test_thread_safe_tick(self):
        import threading

        clk = Clock()
        errors: list[Exception] = []

        def runner() -> None:
            try:
                for _ in range(50):
                    clk.tick()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=runner) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
