"""Tests for GossipState and PeerEndpoint."""

from membrane.network.gossip_state import GossipState, PeerEndpoint


class TestGossipState:
    """Test suite for gossip state serialization and merging."""

    def test_peer_endpoint_roundtrip(self):
        ep = PeerEndpoint(node_id="n1", host="127.0.0.1", port=8080, healthy=True)
        data = ep.to_json()
        restored = PeerEndpoint.from_json(data)
        assert restored.node_id == "n1"
        assert restored.host == "127.0.0.1"
        assert restored.port == 8080
        assert restored.healthy is True

    def test_gossip_state_roundtrip(self):
        state = GossipState(
            node_id="n1",
            timestamp=1234.0,
            peers=[PeerEndpoint("n2", "127.0.0.2", 8081)],
            fragment_locations={"h1": ["n1", "n2"]},
            inventory_digest={"h1": 1},
        )
        data = state.to_json()
        restored = GossipState.from_json(data)
        assert restored.node_id == "n1"
        assert restored.timestamp == 1234.0
        assert len(restored.peers) == 1
        assert restored.fragment_locations == {"h1": ["n1", "n2"]}
        assert restored.inventory_digest == {"h1": 1}

    def test_merge_combines_peers(self):
        a = GossipState(
            node_id="n1",
            timestamp=1000.0,
            peers=[PeerEndpoint("n2", "127.0.0.2", 8081, healthy=True)],
        )
        b = GossipState(
            node_id="n3",
            timestamp=2000.0,
            peers=[PeerEndpoint("n4", "127.0.0.4", 8083, healthy=True)],
        )
        merged = a.merge(b)
        node_ids = {p.node_id for p in merged.peers}
        assert node_ids == {"n2", "n4"}
        assert merged.timestamp == 2000.0

    def test_merge_prefers_healthy_peer(self):
        a = GossipState(
            node_id="n1",
            timestamp=1000.0,
            peers=[PeerEndpoint("n2", "127.0.0.2", 8081, healthy=False)],
        )
        b = GossipState(
            node_id="n3",
            timestamp=2000.0,
            peers=[PeerEndpoint("n2", "127.0.0.2", 8081, healthy=True)],
        )
        merged = a.merge(b)
        peer = merged.peers[0]
        assert peer.node_id == "n2"
        assert peer.healthy is True

    def test_merge_combines_fragment_locations(self):
        a = GossipState(
            node_id="n1",
            timestamp=1000.0,
            fragment_locations={"h1": ["n1"]},
        )
        b = GossipState(
            node_id="n2",
            timestamp=2000.0,
            fragment_locations={"h1": ["n2"]},
        )
        merged = a.merge(b)
        assert set(merged.fragment_locations["h1"]) == {"n1", "n2"}
