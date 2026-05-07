"""Tests for ToolTrace memory object."""

from membrane.fragmentation_engine import compute_content_hash
from membrane.tool_trace import ToolTrace


def test_tool_trace_creation():
    t = ToolTrace(
        tool_name="calculator",
        input_hash="ih123",
        output_hash="oh123",
        structured_output='{"result": 42}',
        content_hash=compute_content_hash((42,)),
        semantic_hash="sh",
        size_bytes=64,
        reuse_score=0.6,
    )
    assert t.tool_name == "calculator"
    assert t.structured_output == '{"result": 42}'


def test_tool_trace_materialize():
    t = ToolTrace(
        tool_name="search",
        input_hash="ih",
        output_hash="oh",
        structured_output="abc",
        content_hash="ch",
        semantic_hash="sh",
        size_bytes=32,
        reuse_score=0.5,
    )
    frag = t.materialize()
    assert frag.content_hash == "ch"


def test_tool_trace_from_fragment():
    t = ToolTrace(
        tool_name="t",
        input_hash="i",
        output_hash="o",
        structured_output="x",
        content_hash="ch",
        semantic_hash="sh",
        size_bytes=16,
        reuse_score=0.3,
    )
    frag = t.materialize()
    recon = ToolTrace.from_fragment(frag)
    assert recon.content_hash == "ch"
    assert recon.size_bytes == 16
