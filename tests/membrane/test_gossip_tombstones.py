"""Tests for GossipState tombstone merge + Gossip delivery."""

from __future__ import annotations

import json
import threading
import time

import pytest

from membrane.gc import Tombstone, TombstoneTable
from membrane.network.config import ClusterConfig
from membrane.network.gossip import Gossip, GossipState, PeerEndpoint
from membrane.network.membership import Membership
from membrane.registry import Registry
from membrane.ring import Ring
from membrane.shard import Shard
from tests.conftest import make_fragment


def _cfg(node_id: str = "node") -> ClusterConfig:
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


class TestGossipStateTombstones:
    def test_round_trip_preserves_tombstones(self):
        state = GossipState(
            node_id="n1",
            timestamp=1.0,
            fragment_tombstones={"h1": 10.0, "h2": 20.0},
        )
        round_trip = GossipState.from_json(json.loads(json.dumps(state.to_json())))
        assert round_trip.fragment_tombstones == state.fragment_tombstones

    def test_merge_keeps_longer_until(self):
        older = GossipState(node_id="n1", timestamp=1.0, fragment_tombstones={"h1": 5.0})
        newer = GossipState(node_id="n2", timestamp=2.0, fragment_tombstones={"h1": 50.0})
        merged = older.merge(newer)
        assert merged.fragment_tombstones["h1"] == 50.0

    def test_merge_adopts_incoming_for_unknown(self):
        local = GossipState(node_id="n1", timestamp=1.0)
        incoming = GossipState(node_id="n2", timestamp=2.0, fragment_tombstones={"h1": 30.0})
        merged = local.merge(incoming)
        assert merged.fragment_tombstones["h1"] == 30.0


class TestGossipHandleTombstones:
    def test_handle_stamps_incoming_tombstone(self):
        node = _FakeNode()
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("local", ring, shard)
        tomb = TombstoneTable()
        gossip = Gossip(
            membership=mem,
            node=node,
            config=_cfg("local"),
            directory=Registry(),
            tombstones=tomb,
            stop_event=threading.Event(),
            running=[False],
        )
        incoming = GossipState(
            node_id="remote",
            timestamp=time.time(),
            fragment_tombstones={"h1": time.time() + 60.0},
        )
        gossip.handle(incoming.to_json())
        record = tomb.get("h1")
        assert record is not None
        assert "remote" in record.nodes

    def test_build_state_includes_active_tombstones(self):
        node = _FakeNode()
        ring = Ring()
        shard = Shard(ring)
        mem = Membership("local", ring, shard)
        tomb = TombstoneTable()
        # One active, one already expired.
        tomb.record("active", time.time() + 30.0, node_ids={"n1"})
        tomb.record("expired", time.time() - 5.0, node_ids={"n1"})
        gossip = Gossip(
            membership=mem,
            node=node,
            config=_cfg("local"),
            directory=Registry(),
            tombstones=tomb,
            stop_event=threading.Event(),
            running=[False],
        )
        state = gossip.build_state()
        assert "active" in state.fragment_tombstones
        assert "expired" not in state.fragment_tombstones


class _FakeNode:
    """Minimal node stand-in: ``fragments`` is enough for ``build_state``."""

    def __init__(self) -> None:
        self.node_id = "local"
        self.fragments: dict[str, object] = {}

    def get_stats(self) -> object:
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


# Smoke: importing PeerEndpoint / GossipState from the module.


def test_module_exports_present():
    assert {n for n in __import__("membrane.network.gossip", fromlist=["*"]).__all__} >= {
        "Gossip",
        "GossipState",
        "PeerEndpoint",
    }
