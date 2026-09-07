"""Tests for the Bloom + Merkle inventory exchange (Phase 5)."""

from __future__ import annotations

import base64
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from membrane.bloom import BloomFilter
from membrane.gc import TombstoneTable
from membrane.merkle import MerkleTree
from membrane.network.gossip import Gossip, GossipState, PeerEndpoint
from membrane.network.membership import Membership
from membrane.registry import Registry
from membrane.ring import Ring
from membrane.shard import Shard
from tests.conftest import make_fragment


class TestBloomFilter:
    def test_round_trip_empty(self):
        bf = BloomFilter.tuned_for(1000, fp_rate=0.001)
        buf = bf.serialize()
        bf2 = BloomFilter.deserialize(buf)
        assert bf2.m_bits == bf.m_bits
        assert bf2.k_hashes == bf.k_hashes

    def test_add_then_contains(self):
        bf = BloomFilter.tuned_for(1000, fp_rate=0.001).add("hello")
        buf = bf.serialize()
        bf2 = BloomFilter.deserialize(buf)
        assert "hello" in bf2
        assert "missing" not in bf2

    def test_no_false_negatives(self):
        bf = BloomFilter.tuned_for(1000, fp_rate=0.001)
        for i in range(100):
            bf = bf.add(f"item-{i}")
        buf = bf.serialize()
        bf2 = BloomFilter.deserialize(buf)
        for i in range(100):
            assert f"item-{i}" in bf2

    def test_false_positive_rate_within_target(self):
        bf = BloomFilter.tuned_for(1000, fp_rate=0.001)
        for i in range(1000):
            bf = bf.add(f"item-{i}")
        fp = 0
        for i in range(10_000):
            if f"random-{i}" in bf:
                fp += 1
        # FP rate is approximate; 0.005 is a generous bound for a
        # 0.1% target on 10k probes against a 1k-item filter.
        assert fp / 10_000 < 0.005

    def test_immutability(self):
        bf = BloomFilter.tuned_for(10, fp_rate=0.01)
        bf2 = bf.add("h1")
        # Original filter is unchanged.
        assert "h1" not in bf
        # The new filter is different.
        assert "h1" in bf2

    def test_membership(self):
        bf = BloomFilter.tuned_for(10, fp_rate=0.01).add("a").add("b")
        assert "a" in bf
        assert "b" in bf
        assert "c" not in bf

    def test_short_tuned_for_does_not_blow_up(self):
        # Edge case: very small expected_items + high fp.
        bf = BloomFilter.tuned_for(1, fp_rate=0.1)
        assert bf.m_bits > 0

    def test_serialize_magic_mismatch_raises(self):
        with pytest.raises(ValueError, match="magic"):
            BloomFilter.deserialize(b"XXXX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


class TestMerkleTree:
    def test_round_trip(self):
        pairs = [("hash-c", "n1"), ("hash-a", "n2"), ("hash-b", "n1")]
        tree = MerkleTree.from_inventory(pairs)
        buf = json.dumps({"root": tree.root.hex()}).encode()
        restored = MerkleTree.from_inventory(pairs)
        assert restored.root == tree.root
        assert json.loads(buf)["root"] == tree.root.hex()

    def test_empty_has_deterministic_root(self):
        e1 = MerkleTree.from_inventory([])
        e2 = MerkleTree.from_inventory(None)
        assert e1.root == e2.root

    def test_diff_returns_indices_for_added_and_changed(self):
        a = MerkleTree.from_inventory([("h1", "n1"), ("h2", "n1"), ("h3", "n1")])
        # b: h1 changed owner, h2 same, h3 missing.
        b = MerkleTree.from_inventory([("h1", "n2"), ("h2", "n1")])
        indices = a.diff(b)
        # h1 has changed (index 0) and h3 is missing on the other side (index 2).
        assert indices == [0, 2]

    def test_diff_returns_empty_when_in_sync(self):
        a = MerkleTree.from_inventory([("h1", "n1"), ("h2", "n2")])
        b = MerkleTree.from_inventory([("h1", "n1"), ("h2", "n2")])
        assert a.diff(b) == []


def _cfg(node_id: str = "local"):
    from membrane.network.config import ClusterConfig
    return ClusterConfig(
        node_id=node_id,
        peers=[],
        enable_gossip=False,
        enable_replication=False,
        heartbeat_interval_sec=60.0,
        heartbeat_timeout_sec=120.0,
        gossip_interval_sec=60.0,
        retry_delay_sec=0.05,
        max_retries=1,
    )


class _FakeNode:
    def __init__(self, fragments: dict | None = None) -> None:
        self.node_id = "local"
        self.fragments = fragments or {}

    def get_stats(self):
        from dataclasses import dataclass

        @dataclass
        class _S:
            fragment_count: int = 0
            memory_used_bytes: int = 0
            memory_limit_bytes: int = 1
            primary_count: int = 0

        return _S()

    def heartbeat(self) -> float:
        return 0.0


class TestGossipStatePhase5:
    def test_inventory_round_trip(self):
        state = GossipState(
            node_id="n1",
            timestamp=1.0,
            inventory_bloom=b"\x00\x00\x00",
            inventory_merkle_root=b"\x01" * 32,
            inventory_size=42,
        )
        round_trip = GossipState.from_json(json.loads(json.dumps(state.to_json())))
        assert round_trip.inventory_bloom == state.inventory_bloom
        assert round_trip.inventory_merkle_root == state.inventory_merkle_root
        assert round_trip.inventory_size == state.inventory_size

    def test_to_json_base64_encodes_bloom(self):
        state = GossipState(
            node_id="n1",
            timestamp=1.0,
            inventory_bloom=b"\x01\x02\x03",
            inventory_merkle_root=b"\xab" * 32,
            inventory_size=1,
        )
        data = state.to_json()
        # base64(b"\x01\x02\x03") == "AQID"
        assert data["inventory_bloom"] == base64.b64encode(b"\x01\x02\x03").decode("ascii")
        assert data["inventory_merkle_root"] == ("ab" * 32)

    def test_legacy_payload_without_bloom_loads(self):
        data = {
            "node_id": "n1",
            "timestamp": 1.0,
            "peers": [],
            "fragment_locations": {},
            "fragment_tombstones": {},
        }
        # 1.0.x-shaped payload. Should parse with empty bloom + root.
        state = GossipState.from_json(data)
        assert state.inventory_bloom == b""
        assert state.inventory_merkle_root == b""
        assert state.inventory_size == 0


class TestGossipBuildStatePhase5:
    def test_build_state_includes_bloom_and_merkle(self):
        node = _FakeNode({"h1": MagicMock(), "h2": MagicMock()})
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("local", ring, shard)
        gossip = Gossip(
            membership=mem,
            node=node,
            config=_cfg(),
            directory=Registry(),
            tombstones=TombstoneTable(),
            stop_event=threading.Event(),
            running=[False],
        )
        state = gossip.build_state()
        # Bloom filter is non-empty.
        assert len(state.inventory_bloom) > 0
        # Merkle root is 32 bytes.
        assert len(state.inventory_merkle_root) == 32
        # inventory_size == 2.
        assert state.inventory_size == 2
