"""Tests for Peer HTTP client.

The transport is mocked via ``unittest.mock`` directly on the
``Peer.transport`` attribute; no separate stub class is needed.
"""

from unittest.mock import MagicMock

import pytest

from membrane.fragment import Fragment
from membrane.network.peer import Peer
from membrane.serialization import to_dict
from membrane.signature import Signature


def make_fragment(content_hash="abc", embedding=(0.1, 0.2, 0.3)):
    return Fragment(
        content_hash=content_hash,
        embedding=embedding,
        structural_signature=Signature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=10,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestPeerClient:
    """Test suite for the Peer client."""

    def test_heartbeat_success(self):
        """Heartbeat returns the parsed response body."""
        transport = MagicMock()
        transport.request.return_value = {"node_id": "n1", "healthy": True}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        resp = peer.heartbeat()
        assert resp == {"node_id": "n1", "healthy": True}
        transport.request.assert_called_once()
        kwargs = transport.request.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == "http://peer:8080/heartbeat"

    def test_heartbeat_retry_then_fail(self):
        """Heartbeat retries until exhausted, then returns None on persistent failure."""
        transport = MagicMock()
        transport.request.return_value = None
        peer = Peer("http://peer:8080", transport=transport, max_retries=2, retry_delay_sec=0.001)
        resp = peer.heartbeat()
        assert resp is None
        assert transport.request.call_count == 2

    def test_store_fragment_success(self):
        """store_fragment returns True when the peer confirms the store."""
        transport = MagicMock()
        transport.request.return_value = {"success": True, "content_hash": "abc"}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.store_fragment(make_fragment("abc")) is True
        transport.request.assert_called_once()

    def test_retrieve_fragment_success(self):
        """retrieve_fragment returns the fragment when found."""
        frag = make_fragment("r1")
        transport = MagicMock()
        transport.request.return_value = {"found": True, "fragment": to_dict(frag)}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        result = peer.retrieve_fragment("r1")
        assert result is not None
        assert result.content_hash == "r1"

    def test_retrieve_fragment_not_found(self):
        """retrieve_fragment returns None when peer reports missing."""
        transport = MagicMock()
        transport.request.return_value = {"found": False, "fragment": None}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.retrieve_fragment("missing") is None

    def test_join_cluster(self):
        """join_cluster returns the bootstrap response."""
        transport = MagicMock()
        transport.request.return_value = {"success": True, "peers": []}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        resp = peer.join_cluster("self", "127.0.0.1", 8080)
        assert resp["success"] is True
        assert resp["peers"] == []

    def test_leave_cluster(self):
        """leave_cluster returns True when the peer confirms the leave."""
        transport = MagicMock()
        transport.request.return_value = {"success": True}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.leave_cluster("self") is True

    def test_gossip(self):
        """gossip returns the response body unchanged."""
        transport = MagicMock()
        transport.request.return_value = {"peers": [], "inventory_digest": {}}
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        resp = peer.gossip({"peers": [], "inventory_digest": {}})
        assert resp is not None
        assert resp["peers"] == []
