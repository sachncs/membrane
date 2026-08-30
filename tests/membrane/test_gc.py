"""Tests for the GC primitives: RefCount, TombstoneTable, Sweeper."""

from __future__ import annotations

import threading
import time

import pytest

from membrane.gc import RefCount, Sweeper, Tombstone, TombstoneTable


class TestRefCount:
    def test_release_drops_to_zero(self):
        rc = RefCount()
        rc.acquire("h1", "n1")
        assert rc.release("h1", "n1") is True
        assert rc.is_active("h1") is False

    def test_release_keeps_when_other_holders_exist(self):
        rc = RefCount()
        rc.acquire("h1", "n1")
        rc.acquire("h1", "n2")
        assert rc.release("h1", "n1") is False
        assert rc.is_active("h1") is True

    def test_release_unknown_hash_returns_true(self):
        """Releasing an unknown hash is a no-op and signals \"already gone\"."""
        rc = RefCount()
        assert rc.release("never", "n1") is True

    def test_holders_returns_copy(self):
        rc = RefCount()
        rc.acquire("h1", "n1")
        rc.acquire("h1", "n2")
        holders = rc.holders("h1")
        holders.add("n3")
        # The internal state must not be affected by the copy.
        assert "n3" not in rc.holders("h1")

    def test_forget_clears_state(self):
        rc = RefCount()
        rc.acquire("h1", "n1")
        rc.forget("h1")
        assert rc.is_active("h1") is False
        rc.forget("h1")  # idempotent

    def test_total_counts_distinct_hashes(self):
        rc = RefCount()
        assert rc.total() == 0
        rc.acquire("h1", "n1")
        rc.acquire("h1", "n2")
        rc.acquire("h2", "n1")
        assert rc.total() == 2

    def test_thread_safe(self):
        rc = RefCount()
        errors: list[Exception] = []

        def worker(prefix: str) -> None:
            try:
                for i in range(100):
                    rc.acquire(f"{prefix}-{i}", "n1")
                    rc.release(f"{prefix}-{i}", "n1")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestTombstoneTable:
    def test_record_and_get(self):
        table = TombstoneTable()
        table.record("h1", until=time.time() + 5.0)
        assert table.is_active("h1") is True
        assert table.get("h1").content_hash == "h1"

    def test_sweep_expired(self):
        table = TombstoneTable()
        table.record("h1", until=time.time() - 1.0)
        table.record("h2", until=time.time() + 30.0)
        expired = table.sweep_expired()
        assert expired == ["h1"]
        assert table.is_active("h1") is False
        assert table.is_active("h2") is True

    def test_record_extends_until(self):
        table = TombstoneTable()
        table.record("h1", until=time.time() + 1.0)
        table.record("h1", until=time.time() + 30.0)
        assert table.get("h1").until > time.time() + 10.0

    def test_get_returns_none_for_expired(self):
        table = TombstoneTable()
        table.record("h1", until=time.time() - 1.0)
        assert table.get("h1") is None

    def test_clear(self):
        table = TombstoneTable()
        table.record("h1", until=time.time() + 1.0)
        table.record("h2", until=time.time() + 1.0)
        table.clear()
        assert table.total() == 0


class TestTombstoneDataclass:
    def test_frozen(self):
        t = Tombstone(content_hash="h1", until=0.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            t.content_hash = "x"  # type: ignore[misc]


import dataclasses


class TestSweeper:
    def test_start_and_stop(self):
        sweeper = Sweeper(interval_sec=0.05)
        sweeper.start()
        assert sweeper.thread is not None and sweeper.thread.is_alive()
        sweeper.stop()
        assert sweeper.thread is None

    def test_run_once_invokes_callbacks(self):
        sweeper = Sweeper(interval_sec=0.5)
        evicted: list[str] = []

        def evict() -> list[str]:
            return ["h1", "h2"]

        def on_evict(items: list[str]) -> None:
            evicted.extend(items)

        sweeper.on_evict_expired = on_evict
        sweeper.run_once(evict_expired=evict)
        assert evicted == ["h1", "h2"]

    def test_run_once_sweeps_tombstones(self):
        sweeper = Sweeper(interval_sec=0.5)
        table = TombstoneTable()
        table.record("old", until=time.time() - 1.0)
        table.record("new", until=time.time() + 30.0)
        expired: list[str] = []

        def on_tomb(items: list[str]) -> None:
            expired.extend(items)

        sweeper.on_tombstones_expired = on_tomb
        sweeper.run_once(tombstones=table)
        assert expired == ["old"]
        assert table.is_active("new")

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValueError):
            Sweeper(interval_sec=0)
