"""Seed bootstrap must not count joining ourselves as success."""

from __future__ import annotations

from unittest.mock import patch

from membrane.network.membership import Membership
from membrane.ring import Ring
from membrane.shard import Shard


def _membership() -> Membership:
    return Membership("n0", Ring(), Shard())


def test_self_only_seed_is_not_success() -> None:
    m = _membership()
    self_only = {"success": True, "peers": [{"node_id": "n0", "host": "n0.local", "port": 8080}]}
    with patch("membrane.network.membership.Peer") as peer_cls:
        peer_cls.return_value.join_cluster.return_value = self_only
        assert m.join_seeds(["n0.local:8080"], local_node_id="n0", host="n0.local", port=8080) is False
    assert m.peers == {}


def test_falls_through_to_next_seed() -> None:
    m = _membership()
    responses = [
        {"success": True, "peers": [{"node_id": "n0", "host": "n0.local", "port": 8080}]},
        {"success": True, "peers": [{"node_id": "n1", "host": "n1.local", "port": 8080}]},
    ]
    with patch("membrane.network.membership.Peer") as peer_cls:
        peer_cls.return_value.join_cluster.side_effect = responses
        assert m.join_seeds(["n0.local:8080", "n1.local:8080"], local_node_id="n0", host="n0.local", port=8080)
    assert set(m.peers) == {"n1"}
