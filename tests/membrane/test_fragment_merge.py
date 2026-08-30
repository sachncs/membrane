"""Tests for Fragment.merge (max-HLC AP conflict resolution, 2.0+).

At 2.0 the merge rule switched from max(version_id) to max(hlc)
because every fragment carries a hybrid logical clock at write
time. version_id remains as a non-authoritative counter for the
TombstoneTable TTL math but no longer drives convergence.
"""

import pytest

from tests.conftest import make_fragment


def make(content_hash: str, hlc: int, version_id: int = 1):
    return make_fragment(content_hash, version_id=version_id).__class__(
        identity=make_fragment(content_hash).__dict__["identity"],
        payload_ref=make_fragment(content_hash).__dict__["payload_ref"],
        payload_size=100,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=version_id,
        hlc=hlc,
    ) if False else _frag(content_hash, hlc, version_id)


def _frag(content_hash: str, hlc: int, version_id: int = 1):
    from membrane.fragment import Fragment
    from membrane.identity import PayloadIdentity

    f = make_fragment(content_hash)
    ident = f.identity
    return Fragment(
        identity=ident,
        payload_ref=f.payload_ref,
        payload_size=f.payload_size,
        ttl=f.ttl,
        reuse_score=f.reuse_score,
        version_id=version_id,
        hlc=hlc,
    )


def test_merge_higher_hlc_wins():
    """Fragment.merge returns the higher hlc."""
    a = _frag("h", 100)
    b = _frag("h", 500)
    assert a.merge(b).hlc == 500
    assert b.merge(a).hlc == 500


def test_merge_equal_hlc_returns_self():
    """On tie, self wins for determinism."""
    a = _frag("h", 300)
    b = _frag("h", 300)
    assert a.merge(b) is a


def test_merge_rejects_different_identity():
    """Mismatched payload_hash raises ValueError."""
    a = _frag("h1", 100)
    b = _frag("h2", 200)
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
