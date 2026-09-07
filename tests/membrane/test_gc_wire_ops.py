"""Tests for delete / tombstone / purge wire ops and peer forwarding."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from membrane.gc import TombstoneTable
from membrane.network.peer import Peer
from membrane.node import Node
from membrane.transport.ops import op_delete, op_purge, op_tombstone
from tests.conftest import make_fragment


class TestOpsDelete:
    def test_delete_removes_fragment_and_writes_tombstone(self):
        node = Node("n1", max_memory_bytes=10_000)
        tombs = TombstoneTable()
        frag = make_fragment("h1")
        assert node.store(frag, is_primary=True) is True
        status, body = op_delete(node, tombs, "h1", "n1")
        assert status == 200
        assert body["success"] is True
        assert "h1" not in node.fragments
        assert tombs.is_active("h1") is True

    def test_delete_missing_is_a_noop_success(self):
        node = Node("n1", max_memory_bytes=10_000)
        tombs = TombstoneTable()
        status, body = op_delete(node, tombs, "missing", "n1")
        assert status == 200
        assert body["success"] is True
        assert body["noop"] is True

    def test_delete_without_tombstone_table_is_a_hard_delete(self):
        node = Node("n1", max_memory_bytes=10_000)
        frag = make_fragment("h2")
        assert node.store(frag, is_primary=True) is True
        status, body = op_delete(node, None, "h2", "n1")
        assert status == 200
        assert body["success"] is True
        assert "h2" not in node.fragments

    def test_delete_with_explicit_deadline(self):
        node = Node("n1", max_memory_bytes=10_000)
        tombs = TombstoneTable()
        frag = make_fragment("h3")
        assert node.store(frag, is_primary=True) is True
        deadline = time.time() + 5.0
        op_delete(node, tombs, "h3", "n1", tombstone_until=deadline)
        record = tombs.get("h3")
        assert record is not None
        assert abs(record.until - deadline) < 0.5


class TestOpsTombstone:
    def test_record_only(self):
        tombs = TombstoneTable()
        status, body = op_tombstone(tombs, "h1", time.time() + 30.0, "n1")
        assert status == 200
        assert body["success"] is True
        assert tombs.is_active("h1") is True

    def test_record_without_table_errors(self):
        status, body = op_tombstone(None, "h1", time.time() + 30.0, "n1")
        assert status == 200
        assert "error" in body


class TestOpsPurge:
    def test_purge_unconditionally_removes(self):
        node = Node("n1", max_memory_bytes=10_000)
        tombs = TombstoneTable()
        frag = make_fragment("h1")
        assert node.store(frag, is_primary=True) is True
        status, body = op_purge(node, tombs, "h1")
        assert status == 200
        assert body["success"] is True
        assert "h1" not in node.fragments


class TestPeerDelete:
    def test_request_delete_forwards(self):
        transport = MagicMock()
        transport.request.return_value = {"success": True}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.request_delete("h1", "n1") is True
        kwargs = transport.request.call_args.kwargs
        assert kwargs["method"] == "POST"
        body = kwargs["body"]
        import json as _json

        decoded = _json.loads(body.decode())
        assert decoded["content_hash"] == "h1"
        assert decoded["node_id"] == "n1"

    def test_request_delete_returns_false_on_failure(self):
        transport = MagicMock()
        transport.request.return_value = {"success": False}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.request_delete("h1", "n1") is False

    def test_request_tombstone(self):
        transport = MagicMock()
        transport.request.return_value = {"success": True}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.request_tombstone("h1", time.time() + 30.0, "n1") is True
        import json as _json

        body = _json.loads(transport.request.call_args.kwargs["body"].decode())
        assert body["content_hash"] == "h1"
        assert body["node_id"] == "n1"
        assert "until" in body
