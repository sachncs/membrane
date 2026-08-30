"""Tests for Phase 6 region tags + locality-aware placement."""

from __future__ import annotations

import pytest

from membrane.node import Node, NodeAttributes
from membrane.shard import Shard


class TestNodeAttributes:
    def test_default_attributes(self):
        n = Node("n1")
        assert n.attributes.region == "default"
        assert n.attributes.zone == "default"
        assert n.attributes.bandwidth_class == 0

    def test_custom_attributes(self):
        attrs = NodeAttributes(region="us-east-1", zone="us-east-1a", bandwidth_class=2)
        n = Node("n1", attributes=attrs)
        assert n.attributes.region == "us-east-1"
        assert n.attributes.bandwidth_class == 2

    def test_to_dict(self):
        attrs = NodeAttributes(region="eu-west-1", zone="eu-west-1c", bandwidth_class=1)
        d = attrs.to_dict()
        assert d == {"region": "eu-west-1", "zone": "eu-west-1c", "bandwidth_class": 1}


class TestShardLocalityScoredAssign:
    def _shard_with(self, regions: dict[str, str], replica_count: int = 2):
        """Build a Shard with one node per region. The ring is seeded by
        explicit add_node calls; regions are attached via
        node_attributes so locality scoring kicks in.
        """
        shard = Shard(replica_count=replica_count, node_attributes={
            nid: NodeAttributes(region=region)
            for nid, region in regions.items()
        })
        for nid in regions:
            shard.add_node(nid)
        return shard

    def test_first_replica_is_cross_region(self):
        # Three regions, replica_count=2: primary + 1 replica.
        # Replica must land in a different region than the primary.
        shard = self._shard_with(
            {"n1": "us-east-1", "n2": "us-east-2", "n3": "eu-west-1"},
            replica_count=2,
        )
        primary, replicas = shard.locality_scored_assign("any-hash")
        # The primary is the consistent-hash ring's first node; replicas
        # include exactly one node in a different region.
        assert len(replicas) == 2
        cross = [r for r in replicas if shard.node_attributes[r].region != shard.node_attributes[primary].region]
        assert len(cross) >= 1

    def test_preferred_region_for_primary(self):
        # When ``primary_region`` is specified, the primary must be
        # in that region.
        shard = self._shard_with(
            {"n1": "us-east-1", "n2": "us-east-2", "n3": "eu-west-1"},
            replica_count=1,
        )
        primary, _replicas = shard.locality_scored_assign("h", primary_region="us-east-1")
        assert shard.node_attributes[primary].region == "us-east-1"

    def test_preferred_region_falls_back_when_no_match(self):
        # When the requested region has no nodes, fall back to any.
        shard = self._shard_with(
            {"n1": "us-east-1", "n2": "us-east-2"},
            replica_count=1,
        )
        primary, _ = shard.locality_scored_assign("h", primary_region="ap-south-1")
        # Just a node, not the requested region. Fallback behavior.
        assert primary in {"n1", "n2"}

    def test_replica_count_honored(self):
        shard = self._shard_with(
            {
                "n1": "us-east-1",
                "n2": "us-east-1",
                "n3": "us-east-2",
                "n4": "eu-west-1",
            },
            replica_count=3,
        )
        primary, replicas = shard.locality_scored_assign("h")
        assert primary not in replicas
        assert len(replicas) == 3

    def test_falls_back_to_ring_when_no_attributes(self):
        # No node_attributes -> deterministic ring walk.
        shard = Shard(replica_count=1)
        shard.add_node("n1")
        shard.add_node("n2")
        primary, replicas = shard.locality_scored_assign("h")
        # Without attributes the function falls back to the same
        # logic as assign_shard / get_replicas.
        assert primary == shard.assign_shard("h")
        assert replicas[0] != primary
