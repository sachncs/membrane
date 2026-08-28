"""Tests for Peer HTTP client.

Exercises the client via a :class:`StubTransport` so no real HTTP
sockets or stdlib patching is involved.
"""

from unittest.mock import MagicMock

import pytest

from membrane.fragment import Fragment
from membrane.serialization import to_dict
from membrane.signature import Signature
from membrane.network.peer import Peer, StubTransport


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
        transport = StubTransport()
        transport.add("GET", "/heartbeat", {"node_id": "n1", "healthy": True})
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        resp = peer.heartbeat()
        assert resp == {"node_id": "n1", "healthy": True}
        assert transport.calls == [("GET", "/heartbeat", None)]

    def test_heartbeat_retry_then_fail(self):
        """Heartbeat retries until exhausted, then returns None on persistent failure."""
        transport = StubTransport()
        # No stub response registered → returns None for every attempt
        peer = Peer("http://peer:8080", transport=transport, max_retries=2, retry_delay_sec=0.001)
        resp = peer.heartbeat()
        assert resp is None
        assert len(transport.calls) == 2

    def test_store_fragment_success(self):
        """store_fragment returns True when the peer confirms the store."""
        transport = StubTransport()
        transport.add("POST", "/store", {"success": True, "content_hash": "abc"})
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.store_fragment(make_fragment("abc")) is True
        assert len(transport.calls) == 1

    def test_retrieve_fragment_success(self):
        """retrieve_fragment returns the fragment when found."""
        frag = make_fragment("r1")
        transport = StubTransport()
        transport.add(
            "GET",
            "/retrieve?content_hash=r1",
            {"found": True, "fragment": to_dict(frag)},
        )
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        result = peer.retrieve_fragment("r1")
        assert result is not None
        assert result.content_hash == "r1"

    def test_retrieve_fragment_not_found(self):
        """retrieve_fragment returns None when peer reports missing."""
        transport = StubTransport()
        transport.add("GET", "/retrieve?content_hash=missing", {"found": False, "fragment": None})
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.retrieve_fragment("missing") is None

    def test_join_cluster(self):
        """join_cluster returns the bootstrap response."""
        transport = StubTransport()
        transport.add("POST", "/join", {"success": True, "peers": []})
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        resp = peer.join_cluster("self", "127.0.0.1", 8080)
        assert resp["success"] is True
        assert resp["peers"] == []

    def test_leave_cluster(self):
        """leave_cluster returns True when the peer confirms the leave."""
        transport = StubTransport()
        transport.add("POST", "/leave", {"success": True})
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        assert peer.leave_cluster("self") is True

    def test_gossip(self):
        """gossip returns the response body unchanged."""
        transport = StubTransport()
        transport.add("POST", "/gossip", {"peers": [], "inventory_digest": {}})
        peer = Peer("http://peer:8080", transport=transport, max_retries=1)
        resp = peer.gossip({"peers": [], "inventory_digest": {}})
        assert resp is not None
        assert resp["peers"] == []
