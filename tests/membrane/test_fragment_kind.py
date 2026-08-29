"""Tests for FragmentKind enum."""

from membrane.fragment_kind import FragmentKind


def test_values_are_wire_format_strings():
    """Enum values must equal the legacy wire strings so serialized fragments stay readable."""
    assert FragmentKind.PREFIX.value == "prefix"
    assert FragmentKind.KV.value == "kv"
    assert FragmentKind.ARTIFACT.value == "artifact"
    assert FragmentKind.TRACE.value == "tool"
    assert FragmentKind.WEIGHTED.value == "weighted_graph"


def test_enum_is_string_subclass():
    """Comparison against raw model_id strings must succeed."""
    assert FragmentKind.PREFIX == "prefix"
    assert FragmentKind.KV == "kv"
    assert FragmentKind.TRACE == "tool"


def test_enum_members_are_unique():
    """No two kinds share a wire string."""
    values = [k.value for k in FragmentKind]
    assert len(values) == len(set(values))