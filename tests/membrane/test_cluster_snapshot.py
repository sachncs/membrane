"""Tests for save_snapshot / load_snapshot on Membership and Shard."""

from __future__ import annotations

import pytest

from membrane.network.membership import Membership, PeerInfo
from membrane.ring import Ring
from membrane.shard import Shard


class TestMembershipSnapshot:
    def test_save_returns_peers(self):
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("n1", ring, shard)
        mem.add("n2", "127.0.0.1", 8081)
        mem.add("n3", "127.0.0.1", 8082)
        snap = mem.save_snapshot()
        ids = sorted(p["node_id"] for p in snap)
        assert ids == ["n2", "n3"]

    def test_load_replaces_peers(self):
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("n1", ring, shard)
        mem.add("n2", "127.0.0.1", 8081)
        # New membership that we will load the snapshot into.
        ring2 = Ring()
        shard2 = Shard(ring2)
        mem2 = Membership("n1", ring2, shard2)
        mem2.load_snapshot(mem.save_snapshot())
        assert "n2" in mem2.peers

    def test_load_drops_existing_peers(self):
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("n1", ring, shard)
        mem.add("n2", "127.0.0.1", 8081)
        mem.add("n3", "127.0.0.1", 8082)
        mem.load_snapshot(
            [
                {
                    "node_id": "n4",
                    "host": "127.0.0.1",
                    "port": 8083,
                    "healthy": True,
                    "suspect": False,
                    "missed_heartbeats": 0,
                    "cluster_epoch": 0,
                }
            ]
        )
        assert "n2" not in mem.peers
        assert "n3" not in mem.peers
        assert "n4" in mem.peers

    def test_load_preserves_cluster_epoch(self):
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("n1", ring, shard)
        mem.load_snapshot(
            [
                {
                    "node_id": "n2",
                    "host": "127.0.0.1",
                    "port": 8081,
                    "cluster_epoch": 42,
                }
            ]
        )
        assert mem.peers["n2"].cluster_epoch == 42


class TestPeerInfoClusterEpoch:
    def test_default_is_zero(self):
        pi = PeerInfo(node_id="n1", host="h", port=8081)
        assert pi.cluster_epoch == 0

    def test_to_json_includes_epoch(self):
        pi = PeerInfo(node_id="n1", host="h", port=8081, cluster_epoch=7)
        assert pi.to_json()["cluster_epoch"] == 7


class TestShardSnapshot:
    def test_save_returns_maps(self):
        shard = Shard()
        shard.primary_map["hash1"] = "node-1"
        shard.replica_map["hash1"] = {"node-2", "node-3"}
        snap = shard.save_snapshot()
        assert snap["primary_map"]["hash1"] == "node-1"
        # replica sets are sorted lists on disk for stable equality.
        assert snap["replica_map"]["hash1"] == ["node-2", "node-3"]

    def test_load_restores_maps(self):
        shard = Shard()
        shard.load_snapshot(
            {
                "primary_map": {"hash1": "node-1"},
                "replica_map": {"hash1": ["node-2", "node-3"]},
            }
        )
        assert shard.primary_map["hash1"] == "node-1"
        assert shard.replica_map["hash1"] == {"node-2", "node-3"}

    def test_load_unknown_shape_is_noop(self):
        shard = Shard()
        shard.primary_map["preexisting"] = "x"
        shard.load_snapshot({"unexpected_key": "ignored"})
        # Missing primary_map / replica_map keys are treated as
        # empty payloads — load_snapshot fully replaces the
        # in-memory state, so the pre-existing entry is dropped.
        assert shard.primary_map == {}
        assert shard.replica_map == {}

    def test_round_trip_equivalence(self):
        shard = Shard()
        shard.primary_map["h1"] = "n1"
        shard.primary_map["h2"] = "n2"
        shard.replica_map["h1"] = {"r1", "r2"}
        snap = shard.save_snapshot()
        fresh = Shard()
        fresh.load_snapshot(snap)
        assert fresh.primary_map == shard.primary_map
        assert fresh.replica_map == shard.replica_map
