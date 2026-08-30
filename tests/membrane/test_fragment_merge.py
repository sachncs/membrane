"""Tests for Fragment.merge (max-version AP conflict resolution)."""

import pytest

from tests.conftest import make_fragment


def make(content_hash: str, version_id: int):
    return make_fragment(content_hash, version_id=version_id)


def test_merge_higher_version_wins():
    """Fragment.merge returns the higher version_id."""
    a = make("h", 1)
    b = make("h", 5)
    assert a.merge(b) is b
    assert b.merge(a) is b


def test_merge_equal_version_returns_self():
    """On tie, self wins for determinism."""
    a = make("h", 3)
    b = make("h", 3)
    assert a.merge(b) is a


def test_merge_rejects_different_payload_hash():
    """Mismatched payload_hash raises ValueError."""
    a = make("h1", 1)
    b = make("h2", 2)
    with pytest.raises(ValueError, match="different identity"):
        a.merge(b)


def test_gossip_state_merge_uses_max_version():
    """GossipState.merge keeps the higher version_id per hash."""
    from membrane.network.gossip import GossipState, PeerEndpoint

    self_state = GossipState(
        node_id="self",
        timestamp=1.0,
        peers=[],
        fragment_locations={},
        inventory_digest={"a": 1, "b": 2},
    )
    other_state = GossipState(
        node_id="other",
        timestamp=2.0,
        peers=[],
        fragment_locations={},
        inventory_digest={"a": 3, "b": 1, "c": 5},
    )
    merged = self_state.merge(other_state)
    # Max wins per key.
    assert merged.inventory_digest == {"a": 3, "b": 2, "c": 5}


def test_gossip_state_merge_does_not_regress():
    """A gossip with older version_id does not overwrite newer local state."""
    from membrane.network.gossip import GossipState

    self_state = GossipState(
        node_id="self",
        timestamp=1.0,
        peers=[],
        fragment_locations={},
        inventory_digest={"a": 10},
    )
    stale_state = GossipState(
        node_id="other",
        timestamp=2.0,
        peers=[],
        fragment_locations={},
        inventory_digest={"a": 5},
    )
    merged = self_state.merge(stale_state)
    assert merged.inventory_digest["a"] == 10
