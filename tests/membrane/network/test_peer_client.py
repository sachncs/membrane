"""Tests for PeerClient."""

from unittest.mock import MagicMock, patch

from membrane.fragment import Fragment
from membrane.network.peer_client import PeerClient
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash: str = "h1"):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 10)
        ),
        size=100,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


def _mock_response(body: bytes):
    """Return a mock that works as a context manager for urlopen."""
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


class TestPeerClient:
    """Test suite for PeerClient HTTP transport."""

    def test_heartbeat_success(self):
        client = PeerClient("http://127.0.0.1:8080")
        mock_resp = _mock_response(b'{"node_id": "n1", "healthy": true}')
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            result = client.heartbeat()
        assert result is not None
        assert result["healthy"] is True

    def test_heartbeat_retry_then_fail(self):
        client = PeerClient("http://127.0.0.1:8080", max_retries=2, retry_delay_sec=0.01)
        with patch("membrane.network.peer_client.urllib.request.urlopen", side_effect=Exception("timeout")):
            result = client.heartbeat()
        assert result is None

    def test_store_fragment_success(self):
        client = PeerClient("http://127.0.0.1:8080")
        frag = make_fragment("abc")
        mock_resp = _mock_response(b'{"success": true, "content_hash": "abc"}')
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            ok = client.store_fragment(frag, is_primary=True)
        assert ok is True

    def test_retrieve_fragment_success(self):
        client = PeerClient("http://127.0.0.1:8080")
        mock_resp = _mock_response(
            b'{"found": true, "fragment": {"content_hash": "abc", "embedding": [0.1], '
            b'"model_id": "m", "layer_range": [0, 1], "token_span": [0, 10], '
            b'"size": 100, "ttl": 3600.0, "reuse_score": 0.5, "version_id": 1}}'
        )
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            frag = client.retrieve_fragment("abc")
        assert frag is not None
        assert frag.content_hash == "abc"

    def test_retrieve_fragment_not_found(self):
        client = PeerClient("http://127.0.0.1:8080")
        mock_resp = _mock_response(b'{"found": false}')
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            frag = client.retrieve_fragment("missing")
        assert frag is None

    def test_join_cluster(self):
        client = PeerClient("http://127.0.0.1:8080")
        mock_resp = _mock_response(b'{"success": true, "peers": []}')
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            result = client.join_cluster("n1", "127.0.0.1", 8080)
        assert result is not None
        assert result["success"] is True

    def test_leave_cluster(self):
        client = PeerClient("http://127.0.0.1:8080")
        mock_resp = _mock_response(b'{"success": true}')
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            ok = client.leave_cluster("n1")
        assert ok is True

    def test_gossip(self):
        client = PeerClient("http://127.0.0.1:8080")
        mock_resp = _mock_response(b'{"node_id": "n2", "timestamp": 1000.0}')
        with patch("membrane.network.peer_client.urllib.request.urlopen", return_value=mock_resp):
            result = client.gossip({"node_id": "n1", "timestamp": 500.0})
        assert result is not None
        assert result["node_id"] == "n2"
