"""Tests for shard_manager module."""

import pytest

from membrane.hash_ring import EmptyRingError, HashRing
from membrane.shard_manager import ShardManager


class TestShardManager:
    """Test suite for ShardManager."""

    def test_assign_shard_on_empty_ring_raises(self):
        sm = ShardManager()
        with pytest.raises(EmptyRingError):
            sm.assign_shard("abc")

    def test_assign_shard_caches_primary(self):
        ring = HashRing()
        ring.add_node("n1")
        sm = ShardManager(hash_ring=ring)
        primary = sm.assign_shard("h1")
        assert primary == "n1"
        assert sm.primary_map["h1"] == "n1"

    def test_get_replicas(self):
        ring = HashRing()
        ring.add_node("n1")
        ring.add_node("n2")
        ring.add_node("n3")
        sm = ShardManager(hash_ring=ring, replica_count=2)
        replicas = sm.get_replicas("h1")
        assert len(replicas) == 2
        assert "n1" not in replicas  # n1 is primary

    def test_get_all_nodes(self):
        ring = HashRing()
        for n in ["a", "b", "c"]:
            ring.add_node(n)
        sm = ShardManager(hash_ring=ring, replica_count=2)
        nodes = sm.get_all_nodes("h1")
        assert len(nodes) == 3

    def test_is_primary(self):
        ring = HashRing()
        ring.add_node("n1")
        sm = ShardManager(hash_ring=ring)
        sm.assign_shard("h1")
        assert sm.is_primary("h1", "n1")
        assert not sm.is_primary("h1", "n2")

    def test_is_replica(self):
        ring = HashRing()
        ring.add_node("n1")
        ring.add_node("n2")
        sm = ShardManager(hash_ring=ring, replica_count=1)
        sm.assign_shard("h1")
        assert sm.is_replica("h1", "n2")
        assert not sm.is_replica("h1", "n1")

    def test_add_node_triggers_rebalance(self):
        ring = HashRing()
        ring.add_node("n1")
        sm = ShardManager(hash_ring=ring)
        sm.assign_shard("h1")
        sm.add_node("n2")
        # After rebalance, h1 may or may not have moved
        assert len(sm.primary_map) == 1

    def test_remove_node_triggers_rebalance(self):
        ring = HashRing()
        ring.add_node("n1")
        ring.add_node("n2")
        sm = ShardManager(hash_ring=ring)
        sm.assign_shard("h1")
        sm.remove_node("n1")
        # After removal, h1 should have a new primary
        assert sm.primary_map["h1"] != "n1"

    def test_shards_for_node(self):
        ring = HashRing()
        ring.add_node("n1")
        sm = ShardManager(hash_ring=ring)
        sm.assign_shard("a")
        sm.assign_shard("b")
        assert sm.shards_for_node("n1") == {"a", "b"}

    def test_replica_shards_for_node(self):
        ring = HashRing()
        ring.add_node("n1")
        ring.add_node("n2")
        sm = ShardManager(hash_ring=ring, replica_count=1)
        sm.assign_shard("h1")
        replicas = sm.replica_shards_for_node("n2")
        assert "h1" in replicas

    def test_rebalance_returns_migrations(self):
        ring = HashRing()
        ring.add_node("n1")
        sm = ShardManager(hash_ring=ring)
        sm.assign_shard("h1")
        ring.add_node("n2")
        migrations = sm.rebalance()
        # May or may not migrate depending on hash
        assert isinstance(migrations, list)

    def test_replica_count_zero(self):
        ring = HashRing()
        ring.add_node("n1")
        sm = ShardManager(hash_ring=ring, replica_count=0)
        replicas = sm.get_replicas("h1")
        assert replicas == set()
