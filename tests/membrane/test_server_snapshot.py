"""Tests for Server snapshot wiring.

Exercises the restore_state / checkpoint_state flow on the
:class:`membrane.server.Server`. The integration paths exercised
here include Membership.load_snapshot, Shard.load_snapshot, the
CheckpointThread daemon, and the cluster_epoch guard rejecting
stale persisted state.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from membrane.network.config import ClusterConfig
from membrane.node import Node
from membrane.server import Server


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


class TestServerCheckpoint:
    """Server.start / stop drive a snapshot round-trip on disk."""

    def test_first_boot_writes_a_snapshot(self, tmp_path: Path):
        node = Node("first", max_memory_bytes=10_000)
        server = Server(
            node=node,
            cluster_config=_cfg("first"),
            port=18181,
            state_dir=str(tmp_path / "state"),
            checkpoint_interval_sec=0,
        )
        try:
            server.start()
            server.checkpoint_state()
            assert (tmp_path / "state" / "first.json").exists()
        finally:
            server.stop()

    def test_restart_hydrates_membership(self, tmp_path: Path):
        state_dir = str(tmp_path / "state")
        node_a = Node("restored", max_memory_bytes=10_000)
        server_a = Server(
            node=node_a,
            cluster_config=_cfg("restored"),
            port=18182,
            state_dir=state_dir,
        )
        try:
            server_a.start()
            server_a.cluster_manager.membership.add("peer-x", "127.0.0.1", 8090)
            server_a.cluster_manager.shard_manager.primary_map["hash-1"] = "restored"
            server_a.checkpoint_state()
        finally:
            server_a.stop()

        node_b = Node("restored", max_memory_bytes=10_000)
        server_b = Server(
            node=node_b,
            cluster_config=_cfg("restored"),
            port=18183,
            state_dir=state_dir,
        )
        try:
            server_b.start()
            assert "peer-x" in server_b.cluster_manager.membership.peers
            assert server_b.cluster_manager.shard_manager.primary_map["hash-1"] == "restored"
        finally:
            server_b.stop()

    def test_stale_epoch_discards_snapshot(self, tmp_path: Path):
        state_dir = str(tmp_path / "state")
        node_a = Node("stale", max_memory_bytes=10_000)
        server_a = Server(
            node=node_a,
            cluster_config=_cfg("stale"),
            port=18184,
            state_dir=state_dir,
            cluster_epoch=10,
        )
        try:
            server_a.start()
            server_a.cluster_manager.membership.add("peer", "127.0.0.1", 8090)
            server_a.checkpoint_state()
        finally:
            server_a.stop()

        node_b = Node("stale", max_memory_bytes=10_000)
        server_b = Server(
            node=node_b,
            cluster_config=_cfg("stale"),
            port=18185,
            state_dir=state_dir,
            cluster_epoch=50,
        )
        try:
            server_b.start()
            assert "peer" not in server_b.cluster_manager.membership.peers
        finally:
            server_b.stop()

    def test_checkpoint_thread_writes_periodically(self, tmp_path: Path):
        node = Node("ticker", max_memory_bytes=10_000)
        server = Server(
            node=node,
            cluster_config=_cfg("ticker"),
            port=18186,
            state_dir=str(tmp_path / "state"),
            checkpoint_interval_sec=0.05,
        )
        try:
            server.start()
            deadline = time.time() + 1.5
            while time.time() < deadline:
                time.sleep(0.05)
                if (tmp_path / "state" / "ticker.json").exists():
                    break
            assert (tmp_path / "state" / "ticker.json").exists()
        finally:
            server.stop()

    def test_no_state_dir_is_a_noop(self):
        node = Node("none", max_memory_bytes=10_000)
        server = Server(
            node=node,
            cluster_config=_cfg("none"),
            port=18187,
            state_dir=None,
        )
        # Both calls must short-circuit cleanly when state_dir
        # was not provided at construction.
        server.restore_state()
        server.checkpoint_state()
