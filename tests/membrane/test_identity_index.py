"""Tests for the IdentityIndex."""

from __future__ import annotations

import pytest

from membrane.identity import PayloadIdentity
from membrane.identity_index import IdentityIndex


def _identity(payload_hash: str = "a" * 64, **overrides: object) -> PayloadIdentity:
    base = dict(
        payload_hash=payload_hash,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 128),
        dtype="float16",
        shape=(1, 1, 1, 128, 64),
    )
    base.update(overrides)
    return PayloadIdentity(**base)  # type: ignore[arg-type]


class TestIdentityIndex:
    """Identity-keyed lookup with collision-safe semantics."""

    def test_round_trip(self):
        idx = IdentityIndex()
        ident = _identity()
        entry = idx.insert(ident, ident.payload_hash)
        assert entry.identity == ident
        assert idx.lookup(ident) == entry

    def test_hash_mismatch_rejected(self):
        idx = IdentityIndex()
        ident = _identity(payload_hash="a" * 64)
        with pytest.raises(ValueError, match="disagrees"):
            idx.insert(ident, "b" * 64)

    def test_different_layer_range_keeps_apart(self):
        """Two variants of the same payload hash remain distinguishable."""
        idx = IdentityIndex()
        ident_a = _identity(layer_range=(0, 1))
        ident_b = _identity(layer_range=(0, 32))
        idx.insert(ident_a, ident_a.payload_hash)
        idx.insert(ident_b, ident_b.payload_hash)
        assert idx.lookup(ident_a).identity.layer_range == (0, 1)
        assert idx.lookup(ident_b).identity.layer_range == (0, 32)
        # reverse index has both fingerprints.
        assert len(idx.lookup_by_hash(ident_a.payload_hash)) == 2

    def test_remove(self):
        idx = IdentityIndex()
        ident = _identity()
        idx.insert(ident, ident.payload_hash)
        assert idx.remove(ident) is True
        assert idx.lookup(ident) is None
        assert idx.lookup_by_hash(ident.payload_hash) == []
        # Removing again is a no-op.
        assert idx.remove(ident) is False

    def test_remove_keeps_other_variant(self):
        idx = IdentityIndex()
        ident_a = _identity(layer_range=(0, 1))
        ident_b = _identity(layer_range=(0, 32))
        idx.insert(ident_a, ident_a.payload_hash)
        idx.insert(ident_b, ident_b.payload_hash)
        idx.remove(ident_a)
        assert idx.lookup(ident_a) is None
        assert idx.lookup(ident_b) is not None
        assert len(idx.lookup_by_hash(ident_a.payload_hash)) == 1

    def test_contains(self):
        idx = IdentityIndex()
        ident = _identity()
        assert ident not in idx
        idx.insert(ident, ident.payload_hash)
        assert ident in idx

    def test_len(self):
        idx = IdentityIndex()
        assert len(idx) == 0
        idx.insert(_identity(), _identity().payload_hash)
        assert len(idx) == 1

    def test_clear(self):
        idx = IdentityIndex()
        ident = _identity()
        idx.insert(ident, ident.payload_hash)
        idx.clear()
        assert len(idx) == 0
        assert ident not in idx

    def test_fingerprint_collides_only_on_every_field_equal(self):
        """Only an exact 10-field match produces the same fingerprint."""
        same = (_identity(), _identity(model_id="m", model_revision=""))
        diff = (
            _identity(),
            _identity(layer_range=(1, 2)),
        )
        assert same[0].fingerprint() == same[1].fingerprint()
        assert diff[0].fingerprint() != diff[1].fingerprint()
