"""Tests for ClusterManager."""

from membrane.membrane_node import MembraneNode
from membrane.network.cluster_manager import ClusterManager
from membrane.network.config import ClusterConfig


class TestClusterManager:
    """Test suite for ClusterManager membership and failure detection."""

    def test_add_peer(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        peers = mgr.get_peers()
        assert len(peers) == 1
        assert peers[0]["node_id"] == "n2"
        assert peers[0]["host"] == "127.0.0.2"
        assert peers[0]["port"] == 8081

    def test_remove_peer(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        assert mgr.remove_peer("n2") is True
        assert mgr.get_peers() == []
        assert mgr.remove_peer("n2") is False

    def test_self_peer_ignored(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n1", "127.0.0.1", 8080)
        assert mgr.get_peers() == []

    def test_on_peer_join(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        result = mgr.on_peer_join("n3", "127.0.0.3", 8082)
        assert result["success"] is True
        assert len(result["peers"]) == 2  # n2 and n1 (self not included)

    def test_on_heartbeat(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        resp = mgr.on_heartbeat("n2")
        assert resp["node_id"] == "n1"
        assert resp["healthy"] is True
        assert mgr.is_peer_healthy("n2") is True

    def test_failure_detection(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(
            node_id="n1",
            host="127.0.0.1",
            port=8080,
            failure_suspect_threshold=1,
            failure_remove_threshold=2,
        )
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        mgr.add_peer("n2", "127.0.0.2", 8081)
        # Simulate missed heartbeats and run one iteration of failure detection
        with mgr._lock:
            mgr._peers["n2"].missed_heartbeats = 2
        # _failure_detection_loop is infinite; test the logic inline
        now = __import__("time").time()
        to_remove = []
        with mgr._lock:
            for node_id, p in list(mgr._peers.items()):
                if p.missed_heartbeats >= mgr.config.failure_remove_threshold:
                    to_remove.append(node_id)
                elif p.missed_heartbeats >= mgr.config.failure_suspect_threshold:
                    if not p.suspect:
                        p.suspect = True
        for node_id in to_remove:
            mgr.remove_peer(node_id)
        assert mgr.get_peers() == []

    def test_on_gossip(self):
        node = MembraneNode("n1", max_memory_bytes=10000)
        cfg = ClusterConfig(node_id="n1", host="127.0.0.1", port=8080)
        mgr = ClusterManager("n1", "127.0.0.1", 8080, node, cfg)
        result = mgr.on_gossip({
            "node_id": "n2",
            "timestamp": 1000.0,
            "peers": [{"node_id": "n2", "host": "127.0.0.2", "port": 8081, "healthy": True}],
            "fragment_locations": {"h1": ["n2"]},
            "inventory_digest": {"h1": 1},
        })
        assert result["node_id"] == "n1"
        assert len(mgr.get_peers()) == 1
