"""Tests for delta_encoder module."""

import pytest

from membrane.delta_encoder import Delta, DeltaEncoder
from membrane.fragmentation_engine import compute_content_hash


class TestDeltaEncoder:
    """Test suite for DeltaEncoder."""

    def test_encode_identical_returns_empty_delta(self):
        enc = DeltaEncoder()
        base = (1, 2, 3)
        delta = enc.encode(base, base)
        assert delta.removed_tail_count == 0
        assert delta.appended_tokens == ()
        assert delta.base_content_hash == compute_content_hash(base)

    def test_encode_appended_tokens(self):
        enc = DeltaEncoder()
        base = (1, 2, 3)
        new = (1, 2, 3, 4, 5)
        delta = enc.encode(base, new)
        assert delta.appended_tokens == (4, 5)
        assert delta.removed_tail_count == 0

    def test_encode_removed_and_appended(self):
        enc = DeltaEncoder()
        base = (1, 2, 3, 4)
        new = (1, 2, 5)
        delta = enc.encode(base, new)
        assert delta.removed_tail_count == 2
        assert delta.appended_tokens == (5,)

    def test_decode_reconstructs_original(self):
        enc = DeltaEncoder()
        base = (1, 2, 3)
        new = (1, 2, 3, 4)
        delta = enc.encode(base, new)
        reconstructed = enc.decode(base, delta)
        assert reconstructed == new

    def test_decode_with_removal(self):
        enc = DeltaEncoder()
        base = (1, 2, 3, 4, 5)
        new = (1, 2)
        delta = enc.encode(base, new)
        reconstructed = enc.decode(base, delta)
        assert reconstructed == new

    def test_encode_empty_base(self):
        enc = DeltaEncoder()
        base = ()
        new = (1, 2, 3)
        delta = enc.encode(base, new)
        assert delta.removed_tail_count == 0
        assert delta.appended_tokens == (1, 2, 3)

    def test_roundtrip_random(self):
        enc = DeltaEncoder()
        base = tuple(range(10))
        new = tuple(range(7)) + (99, 100)
        delta = enc.encode(base, new)
        assert enc.decode(base, delta) == new
