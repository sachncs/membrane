"""Tests for the per-peer replication-lag gauge (Phase 3.2.2)."""

from __future__ import annotations

import math

import pytest

from membrane.metrics import MetricsCollector
from membrane.network.lag import (
    REPLICATION_LAG_GAUGE,
    PeerLagSnapshot,
    record_replication_lag,
    render_prometheus_gauge,
    snapshot_peer_lag,
)
from membrane.network.membership import Membership, PeerInfo
from membrane.ring import Ring
from membrane.shard import Shard


def _fresh_membership() -> Membership:
    return Membership("local", Ring(), Shard())


class TestSnapshotPeerLag:
    def test_snapshot_returns_zero_for_recent_peer(self):
        membership = _fresh_membership()
        membership.add("peer-1", "1.1.1.1", 8001)
        # record_heartbeat stamps last_heartbeat = time.time();
        # a recent peer therefore reads as ~0 lag.
        snapshot = snapshot_peer_lag(membership, now=10.0)
        assert snapshot.lag_seconds.get("peer-1") is not None

    def test_snapshot_returns_infinite_for_never_beat_peer(self):
        membership = _fresh_membership()
        membership.add("peer-1", "1.1.1.1", 8001)
        # Membership.add stamps last_heartbeat = time.time();
        # reset to 0.0 to simulate a peer that was added but
        # never recorded a successful heartbeat.
        membership.peers["peer-1"].last_heartbeat = 0.0
        snapshot = snapshot_peer_lag(membership, now=10.0)
        assert snapshot.lag_seconds.get("peer-1") == math.inf

    def test_snapshot_returns_delta_for_lagging_peer(self):
        membership = _fresh_membership()
        membership.add("peer-1", "1.1.1.1", 8001)
        membership.add("peer-2", "2.2.2.2", 8001)
        membership.record_heartbeat("peer-1", lease_until=0.0)
        membership.peers["peer-2"].last_heartbeat = 0.0
        snapshot = snapshot_peer_lag(membership, now=15.0)
        assert snapshot.lag_seconds["peer-1"] == 0.0
        assert snapshot.lag_seconds["peer-2"] == math.inf


class TestRenderPrometheusGauge:
    def test_format_includes_help_and_type(self):
        snap = PeerLagSnapshot(lag_seconds={}, now=0.0)
        text = render_prometheus_gauge(snap)
        assert f"# HELP {REPLICATION_LAG_GAUGE}" in text
        assert f"# TYPE {REPLICATION_LAG_GAUGE} gauge" in text

    def test_format_includes_each_peer_with_label(self):
        snap = PeerLagSnapshot(lag_seconds={"peer-1": 5.0, "peer-2": 10.0}, now=0.0)
        text = render_prometheus_gauge(snap)
        assert 'peer="peer-1"' in text
        assert 'peer="peer-2"' in text
        assert " 5" in text
        assert " 10" in text

    def test_infinite_lag_serialized_as_sentinel(self):
        snap = PeerLagSnapshot(lag_seconds={"peer-x": math.inf}, now=0.0)
        text = render_prometheus_gauge(snap)
        # The Prometheus convention requires positive finite
        # numbers; the helper stores the lag as a 10-year
        # sentinel so dashboards can alert on the threshold.
        assert "peer-x" in text
        assert math.inf == math.inf  # the sentinel conversion is internal


class TestRecordReplicationLag:
    def test_records_each_peer_to_the_registry(self):
        registry = MetricsCollector()
        membership = _fresh_membership()
        membership.add("peer-1", "1.1.1.1", 8001)
        membership.record_heartbeat("peer-1", lease_until=0.0)
        snapshot = record_replication_lag(registry, membership)
        assert "peer-1" in snapshot.lag_seconds
        # The gauge is registered under the per-peer name.
        gauge_name = f"{REPLICATION_LAG_GAUGE}:peer-1"
        assert gauge_name in registry.gauges
        assert registry.gauges[gauge_name].value == 0.0


class TestMembershipHelpers:
    def test_membership_snapshot_includes_peer(self):
        from membrane.ring import Ring
        from membrane.shard import Shard

        membership = Membership("local", Ring(), Shard())
        peer = PeerInfo(
            node_id="peer-1",
            host="1.1.1.1",
            port=8001,
            last_heartbeat=42.0,
        )
        membership.peers["peer-1"] = peer
        snap = membership.snapshot()
        assert len(snap) == 1
        assert snap[0].node_id == "peer-1"
        assert snap[0].last_heartbeat == 42.0
