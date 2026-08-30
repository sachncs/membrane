"""Tests for Phase 4 leases + drain."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from membrane.network.membership import Membership
from membrane.network.peer import Peer
from membrane.ring import Ring
from membrane.shard import Shard
from tests.conftest import make_fragment


class TestLeaseSemantics:
    def test_heartbeat_refresh_lease(self):
        mem = Membership("local", Ring(), Shard())
        mem.add("n2", "127.0.0.1", 8081)
        deadline = time.time() + 60
        mem.record_heartbeat("n2", lease_until=deadline)
        assert mem.peers["n2"].lease_until == deadline
        assert mem.peers["n2"].healthy is True

    def test_heartbeat_zero_lease_keeps_zero(self):
        mem = Membership("local", Ring(), Shard())
        mem.add("n2", "127.0.0.1", 8081)
        mem.record_heartbeat("n2", lease_until=0)
        assert mem.peers["n2"].lease_until == 0

    def test_evict_expired_leases_flags_suspect(self):
        mem = Membership("local", Ring(), Shard())
        mem.add("n2", "127.0.0.1", 8081)
        mem.peers["n2"].lease_until = time.time() - 5
        flagged = mem.evict_expired_leases()
        assert flagged == ["n2"]
        assert mem.peers["n2"].suspect is True

    def test_evict_expired_leases_keeps_fresh(self):
        mem = Membership("local", Ring(), Shard())
        mem.add("n2", "127.0.0.1", 8081)
        mem.peers["n2"].lease_until = time.time() + 60
        flagged = mem.evict_expired_leases()
        assert flagged == []
        assert mem.peers["n2"].suspect is False

    def test_evict_expired_leases_ignores_zero_lease(self):
        """A peer with lease_until=0 means 'no explicit lease'; don't flag."""
        mem = Membership("local", Ring(), Shard())
        mem.add("n2", "127.0.0.1", 8081)
        # lease_until stays at 0
        flagged = mem.evict_expired_leases()
        assert flagged == []
        assert mem.peers["n2"].suspect is False

    def test_snapshot_round_trip_carries_lease(self):
        from membrane.snapshot import Snapshot

        snap = Snapshot("/tmp/membrane-test-snap")  # tmp dir, used in-memory
        # Use a tmp dir provided by pytest.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            snap = Snapshot(tmp)
            mem = Membership("local", Ring(), Shard())
            mem.add("n2", "127.0.0.1", 8081, peer_cn="admin-1")
            mem.peers["n2"].lease_until = 12345.0
            payload = mem.save_snapshot()
            snap.save("local", {"schema_version": 2, "cluster_epoch": 1, "membership": payload, "shards": {}, "server": {}})
            loaded = snap.load("local")
            assert loaded is not None
            mem2 = Membership("local", Ring(), Shard())
            mem2.load_snapshot(loaded["membership"])
            assert mem2.peers["n2"].lease_until == 12345.0


class TestMembershipLeaveCluster:
    def test_leave_dispatches_to_each_peer(self):
        mem = Membership("local", Ring(), Shard())
        for i in range(3):
            mem.add(f"peer-{i}", "127.0.0.1", 8000 + i)
        clients: dict[str, MagicMock] = {}
        for i in range(3):
            client = MagicMock(spec=Peer)
            client.leave_cluster.return_value = True
            clients[f"peer-{i}"] = client

        mem.get_client = lambda nid: clients.get(nid)
        count = mem.leave_cluster(local_node_id="local")
        assert count == 3
        for c in clients.values():
            c.leave_cluster.assert_called_once_with("local")

    def test_leave_skips_failed_peers(self):
        mem = Membership("local", Ring(), Shard())
        mem.add("p1", "127.0.0.1", 8001)
        mem.add("p2", "127.0.0.1", 8002)
        c1 = MagicMock(spec=Peer)
        c1.leave_cluster.side_effect = RuntimeError("network")
        c2 = MagicMock(spec=Peer)
        c2.leave_cluster.return_value = True
        clients = {"p1": c1, "p2": c2}
        mem.get_client = lambda nid: clients.get(nid)
        count = mem.leave_cluster(local_node_id="local")
        assert count == 1
        c2.leave_cluster.assert_called_once_with("local")
