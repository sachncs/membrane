"""Tests for Registry.forget_fragment_location + forget_fragment."""

from __future__ import annotations

from membrane.registry import Registry


class TestRegistryForget:
    def test_record_then_forget_one_node(self):
        r = Registry()
        r.record_fragment_location("h1", "n1")
        r.record_fragment_location("h1", "n2")
        assert r.locate_fragment("h1") == {"n1", "n2"}
        assert r.forget_fragment_location("h1", "n1") is True
        assert r.locate_fragment("h1") == {"n2"}

    def test_forget_drops_empty_entry(self):
        r = Registry()
        r.record_fragment_location("h1", "n1")
        r.forget_fragment_location("h1", "n1")
        assert "h1" not in r.fragment_locations

    def test_forget_unknown_returns_false(self):
        r = Registry()
        assert r.forget_fragment_location("missing", "n1") is False
        assert r.forget_fragment_location("h1", "missing") is False

    def test_forget_fragment_drops_every_holder(self):
        r = Registry()
        r.record_fragment_location("h1", "n1")
        r.record_fragment_location("h1", "n2")
        assert r.forget_fragment("h1") is True
        assert r.locate_fragment("h1") == set()
        assert "h1" not in r.fragment_locations

    def test_forget_fragment_missing(self):
        r = Registry()
        assert r.forget_fragment("never") is False

    def test_forget_then_record_round_trips(self):
        """After forget, recording again starts fresh."""
        r = Registry()
        r.record_fragment_location("h1", "n1")
        r.forget_fragment("h1")
        r.record_fragment_location("h1", "n3")
        assert r.locate_fragment("h1") == {"n3"}
