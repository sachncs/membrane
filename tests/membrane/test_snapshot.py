"""Tests for durable snapshot + ClusterEpochGuard."""

from __future__ import annotations

import json

import pytest

from membrane.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    ClusterEpochGuard,
    Snapshot,
)


@pytest.fixture
def snapshot_dir(tmp_path):
    return tmp_path / "state"


class TestSnapshot:
    def test_save_and_load_round_trip(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        payload = {"schema_version": SNAPSHOT_SCHEMA_VERSION, "cluster_epoch": 7, "foo": "bar"}
        snap.save("node-1", payload)
        loaded = snap.load("node-1")
        assert loaded is not None
        assert loaded["foo"] == "bar"
        assert loaded["cluster_epoch"] == 7
        assert loaded["captured_at"] > 0

    def test_load_missing_returns_none(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        assert snap.load("never-saved") is None

    def test_load_wrong_schema_version_returns_none(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        snap.state_dir.mkdir(parents=True, exist_ok=True)
        (snap.state_dir / "node-1.json").write_bytes(
            json.dumps({"schema_version": 1, "cluster_epoch": 0}).encode("utf-8")
        )
        assert snap.load("node-1") is None

    def test_latest_picks_most_recent(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        snap.save("a", {"schema_version": 2, "cluster_epoch": 1, "captured_at": 1.0})
        snap.save("b", {"schema_version": 2, "cluster_epoch": 1, "captured_at": 5.0})
        snap.save("c", {"schema_version": 2, "cluster_epoch": 1, "captured_at": 3.0})
        latest = snap.latest()
        assert latest is not None
        assert "__path__" in latest
        assert "b.json" in latest["__path__"]

    def test_latest_empty(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        assert snap.latest() is None

    def test_remove_returns_true(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        snap.save("node-1", {"schema_version": 2, "cluster_epoch": 0})
        assert snap.remove("node-1") is True
        # Repeat removal returns False.
        assert snap.remove("node-1") is False

    def test_len_counts_snapshots(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        assert len(snap) == 0
        snap.save("a", {"schema_version": 2, "cluster_epoch": 0})
        snap.save("b", {"schema_version": 2, "cluster_epoch": 0})
        assert len(snap) == 2

    def test_atomic_replace_leaves_no_tmp(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        snap.save("node-1", {"schema_version": 2, "cluster_epoch": 0})
        # No leftover .tmp file survived.
        tmp_files = list(snapshot_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_overwrite_replaces(self, snapshot_dir):
        snap = Snapshot(snapshot_dir)
        snap.save("node-1", {"schema_version": 2, "cluster_epoch": 0, "v": 1})
        snap.save("node-1", {"schema_version": 2, "cluster_epoch": 1, "v": 2})
        loaded = snap.load("node-1")
        assert loaded["v"] == 2
        assert loaded["cluster_epoch"] == 1


class TestClusterEpochGuard:
    def test_accept_within_one(self):
        guard = ClusterEpochGuard(current=5)
        assert guard.accept(persisted=5) is True
        assert guard.accept(persisted=4) is True
        assert guard.accept(persisted=6) is True

    def test_reject_too_far_behind(self):
        guard = ClusterEpochGuard(current=10)
        assert guard.accept(persisted=5) is False

    def test_reject_missing(self):
        guard = ClusterEpochGuard(current=10)
        assert guard.accept(persisted=None) is False

    def test_bump_increments(self):
        guard = ClusterEpochGuard(current=10)
        assert guard.bump() == 11
        assert guard.current == 11
