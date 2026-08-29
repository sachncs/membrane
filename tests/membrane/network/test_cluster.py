"""Tests for Cluster."""

from membrane.fragment import Fragment
from membrane.network.cluster import Cluster
from membrane.network.config import ClusterConfig
from membrane.network.strategy import EagerMigrator, RateLimitedMigrator
from membrane.node import Node
from membrane.signature import Signature


class TestClusterManager:
    """Test suite for Cluster membership and failure detection."""

    def test_add_peer(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        peers = mgr.get_peers()
        assert len(peers) == 1
        assert peers[0]["node_id"] == "n2"
        assert peers[0]["host"] == "127.0.0.2"
        assert peers[0]["port"] == 8081

    def test_remove_peer(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        assert mgr.remove_peer("n2") is True
        assert mgr.get_peers() == []
        assert mgr.remove_peer("n2") is False

    def test_self_peer_ignored(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n1", "127.0.0.1", 8080)
        assert mgr.get_peers() == []

    def test_on_peer_join(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        result = mgr.on_peer_join("n3", "127.0.0.3", 8082)
        assert result["success"] is True
        assert len(result["peers"]) == 2  # n2 and n1 (self not included)

    def test_on_heartbeat(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        resp = mgr.on_heartbeat("n2")
        assert resp["node_id"] == "n1"
        assert resp["healthy"] is True
        assert mgr.is_peer_healthy("n2") is True

    def test_failure_detection(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(
            node_id="n1",
            host="127.0.0.1",
            port=8080,
            failure_suspect_threshold=1,
            failure_remove_threshold=2,
        )
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        # Simulate missed heartbeats by reaching into membership.
        peer = mgr.membership.find("n2")
        assert peer is not None
        peer.missed_heartbeats = 2
        # Use the Failure subsystem's detector directly.
        for p in mgr.membership.snapshot():
            if p.missed_heartbeats >= mgr.config.failure_remove_threshold and mgr.failure.detector.should_remove(
                peer_id=p.node_id,
                peer_missed=p.missed_heartbeats,
                suspect_votes=0,
                healthy_peer_count=len(mgr.membership.healthy()) + 1,
            ):
                mgr.remove_peer(p.node_id)
        assert mgr.get_peers() == []

    def test_on_gossip(self):
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster("n1", "127.0.0.1", 8080, node, cfg)
        result = mgr.on_gossip(
            {
                "node_id": "n2",
                "timestamp": 1000.0,
                "peers": [{"node_id": "n2", "host": "127.0.0.2", "port": 8081, "healthy": True}],
                "fragment_locations": {"h1": ["n2"]},
                "inventory_digest": {"h1": 1},
            }
        )
        assert result["node_id"] == "n1"
        assert len(mgr.get_peers()) == 1

    def test_on_peer_leave_migrates_primaries_to_local(self):
        """When a peer leaves, primaries it owned are migrated to the
        local node by the configured Migrator."""
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster(
            "n1",
            "127.0.0.1",
            8080,
            node,
            cfg,
            migrator=EagerMigrator(),
        )
        # Seed the shard table: leaving peer n2 is the primary of
        # h-mig; n1 is already a replica.
        mgr.shard_manager.primary_map["h-mig"] = "n2"
        mgr.shard_manager.replica_map["h-mig"] = {"n1"}
        # Seed a fragment on the local node so the migrator can
        # promote it.
        node.fragments["h-mig"] = Fragment(
            content_hash="h-mig",
            embedding=(0.0,),
            structural_signature=Signature("m", (0, 1), (0, 0)),
            size=10,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )

        mgr.on_peer_leave("n2")

        # Migrator re-homed the hash: n1 promoted to primary, n2 dropped from replicas.
        assert mgr.shard_manager.primary_map["h-mig"] == "n1"
        assert "n2" not in mgr.shard_manager.replica_map["h-mig"]
        assert "h-mig" in node.primary_hashes

    def test_on_peer_leave_uses_rate_limited_migrator(self):
        """RateLimitedMigrator with max_per_second=10 takes ~0.1s for 1 hash."""
        node = Node("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = Cluster(
            "n1",
            "127.0.0.1",
            8080,
            node,
            cfg,
            migrator=RateLimitedMigrator(max_per_second=10.0),
        )
        mgr.shard_manager.primary_map["h-rate"] = "n2"
        mgr.shard_manager.replica_map["h-rate"] = {"n1"}
        import time

        start = time.monotonic()
        mgr.on_peer_leave("n2")
        elapsed = time.monotonic() - start
        # 1 migration at 10/s = 0.1s sleep; allow some slack.
        assert elapsed >= 0.05
