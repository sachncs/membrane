"""Tests for Shard.migrate_primary with the optional TransferService push."""

from __future__ import annotations

from unittest.mock import MagicMock

from membrane.node import Node
from membrane.shard import Shard
from tests.conftest import make_fragment


class TestMigratePrimaryTransfer:
    def test_no_transfer_service_no_push(self):
        shard = Shard()
        node = Node("local", max_memory_bytes=10_000)
        assert node.store(make_fragment("h1"), is_primary=False) is True
        # Migrator path with no transfer_service attached — table
        # updates happen, no push attempted.
        shard.migrate_primary(
            content_hash="h1",
            leaving_peer="peer-1",
            local_node_id="local",
            node=node,
        )
        assert shard.primary_map["h1"] == "local"
        assert "h1" in node.primary_hashes

    def test_push_calls_transfer_fragment(self):
        shard = Shard()
        node = Node("local", max_memory_bytes=10_000)
        assert node.store(make_fragment("h2"), is_primary=False) is True
        push_target = MagicMock()
        shard.migrate_primary(
            content_hash="h2",
            leaving_peer="peer-1",
            local_node_id="local",
            node=node,
            transfer_service=push_target,
        )
        push_target.transfer_fragment.assert_called_once_with(node, "h2")
        assert shard.primary_map["h2"] == "local"

    def test_push_skipped_when_fragment_not_local(self):
        shard = Shard()
        node = Node("local", max_memory_bytes=10_000)
        push_target = MagicMock()
        shard.migrate_primary(
            content_hash="h3",
            leaving_peer="peer-1",
            local_node_id="local",
            node=node,
            transfer_service=push_target,
        )
        # The bytes are not local; push must NOT be attempted.
        push_target.transfer_fragment.assert_not_called()
        # The table still updates — every migration is a
        # bookkeeping commit even when bytes are absent.
        assert shard.primary_map["h3"] == "local"

    def test_push_skipped_when_local_is_not_target(self):
        shard = Shard()
        node = Node("remoteA", max_memory_bytes=10_000)
        assert node.store(make_fragment("h4"), is_primary=False) is True
        push_target = MagicMock()
        shard.migrate_primary(
            content_hash="h4",
            leaving_peer="peer-1",
            local_node_id="remoteB",  # different from node.node_id
            node=node,
            transfer_service=push_target,
        )
        push_target.transfer_fragment.assert_not_called()

    def test_push_failure_does_not_break_migration(self):
        shard = Shard()
        node = Node("local", max_memory_bytes=10_000)
        assert node.store(make_fragment("h5"), is_primary=False) is True

        def boom(_arg, _hash):
            raise RuntimeError("network glitch")

        push_target = MagicMock()
        push_target.transfer_fragment.side_effect = boom
        shard.migrate_primary(
            content_hash="h5",
            leaving_peer="peer-1",
            local_node_id="local",
            node=node,
            transfer_service=push_target,
        )
        # Table migration still completes; failure is logged but
        # does not raise out of migrate_primary.
        assert shard.primary_map["h5"] == "local"
