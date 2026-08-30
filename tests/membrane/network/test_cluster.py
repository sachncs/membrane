"""Tests for Cluster."""

from membrane.fragment import Fragment
from membrane.network.cluster import Cluster
from membrane.network.config import ClusterConfig
from membrane.network.strategy import EagerMigrator, RateLimitedMigrator
from membrane.node import Node
from membrane.signature import Signature


class TestClusterManager:
    """Test suite for Cluster membership and failure detection."""

    def _mgr(self, **cfg_kwargs) -> tuple[Node, Cluster]:
        migrator = cfg_kwargs.pop("migrator", None)
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080, **cfg_kwargs)
        return node, Cluster("n1", "127.0.0.1", 8080, node, cfg, migrator=migrator)

    def test_add_peer(self):
        _, mgr = self._mgr()
        mgr.membership.add("n2", "127.0.0.2", 8081)
        peers = mgr.membership.to_json()
        assert len(peers) == 1
        assert peers[0]["node_id"] == "n2"
        assert peers[0]["host"] == "127.0.0.2"
        assert peers[0]["port"] == 8081

    def test_remove_peer(self):
        _, mgr = self._mgr()
        mgr.membership.add("n2", "127.0.0.2", 8081)
        assert mgr.membership.remove("n2") is True
        assert mgr.membership.to_json() == []
        assert mgr.membership.remove("n2") is False

    def test_self_peer_ignored(self):
        _, mgr = self._mgr()
        mgr.membership.add("n1", "127.0.0.1", 8080)
        assert mgr.membership.to_json() == []

    def test_on_peer_join(self):
        _, mgr = self._mgr()
        mgr.membership.add("n2", "127.0.0.2", 8081)
        mgr.membership.add("n3", "127.0.0.3", 8082)
        assert len(mgr.membership.to_json()) == 2

    def test_on_heartbeat(self):
        _, mgr = self._mgr()
        mgr.membership.add("n2", "127.0.0.2", 8081)
        mgr.membership.record_heartbeat("n2")
        peer = mgr.membership.find("n2")
        assert peer is not None
        assert peer.healthy is True

    def test_failure_detection(self):
        _, mgr = self._mgr(failure_suspect_threshold=1, failure_remove_threshold=2)
        mgr.membership.add("n2", "127.0.0.2", 8081)
        peer = mgr.membership.find("n2")
        assert peer is not None
        peer.missed_heartbeats = 2
        for p in mgr.membership.snapshot():
            if p.missed_heartbeats >= mgr.config.failure_remove_threshold and mgr.failure.detector.should_remove(
                peer_id=p.node_id,
                peer_missed=p.missed_heartbeats,
                suspect_votes=0,
                healthy_peer_count=len(mgr.membership.healthy()) + 1,
            ):
                mgr.membership.remove(p.node_id)
        assert mgr.membership.to_json() == []

    def test_on_gossip(self):
        _, mgr = self._mgr()
        result = mgr.gossip.handle(
            {
                "node_id": "n2",
                "timestamp": 1000.0,
                "peers": [{"node_id": "n2", "host": "127.0.0.2", "port": 8081, "healthy": True}],
                "fragment_locations": {"h1": ["n2"]},
                "inventory_digest": {"h1": 1},
            }
        )
        assert result["node_id"] == "n1"
        assert len(mgr.membership.to_json()) == 1

    def test_on_peer_leave_migrates_primaries_to_local(self):
        """When a peer leaves, primaries it owned are migrated to the
        local node by the configured Migrator."""
        node, mgr = self._mgr(migrator=EagerMigrator())
        mgr.shard_manager.primary_map["h-mig"] = "n2"
        mgr.shard_manager.replica_map["h-mig"] = {"n1"}
        node.fragments["h-mig"] = Fragment(
            content_hash="h-mig",
            embedding=(0.0,),
            structural_signature=Signature("m", (0, 1), (0, 0)),
            size=10,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )

        mgr.membership.remove("n2")
        leaving_hashes = {h for h, primary in mgr.shard_manager.primary_map.items() if primary == "n2"}
        if mgr.migrator is not None and leaving_hashes:
            mgr.migrator.migrate(leaving_hashes, "n2")

        assert mgr.shard_manager.primary_map["h-mig"] == "n1"
        assert "n2" not in mgr.shard_manager.replica_map["h-mig"]
        assert "h-mig" in node.primary_hashes

    def test_on_peer_leave_uses_rate_limited_migrator(self):
        """RateLimitedMigrator with max_per_second=10 takes ~0.1s for 1 hash."""
        import time

        _, mgr = self._mgr(migrator=RateLimitedMigrator(max_per_second=10.0))
        mgr.shard_manager.primary_map["h-rate"] = "n2"
        mgr.shard_manager.replica_map["h-rate"] = {"n1"}

        mgr.membership.remove("n2")
        leaving_hashes = {h for h, primary in mgr.shard_manager.primary_map.items() if primary == "n2"}

        start = time.monotonic()
        mgr.migrator.migrate(leaving_hashes, "n2")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05
