"""Tests for remote_prefill_dispatcher module."""

import pytest

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillAdapter, PrefillResult
from membrane.remote_prefill_dispatcher import RemotePrefillDispatcher
from membrane.structural_signature import StructuralSignature


class TestRemotePrefillDispatcher:
    """Test suite for RemotePrefillDispatcher."""

    def test_dispatch_stores_fragments_on_target(self):
        dispatcher = RemotePrefillDispatcher()
        target = MembraneNode("target")
        tokens = list(range(100))
        result = dispatcher.dispatch(tokens, "model-a", target)
        assert isinstance(result, PrefillResult)
        assert len(result.fragments) > 0
        for frag in result.fragments:
            assert target.retrieve(frag.content_hash) is not None

    def test_dispatch_empty_prompt(self):
        dispatcher = RemotePrefillDispatcher()
        target = MembraneNode("target")
        result = dispatcher.dispatch([], "model-a", target)
        assert result.fragments == []

    def test_custom_adapter(self):
        adapter = PrefillAdapter(compute_scale=0.5)
        dispatcher = RemotePrefillDispatcher(prefill_adapter=adapter)
        assert dispatcher.prefill_adapter.compute_scale == 0.5

    def test_dispatch_returns_kvs_estimate(self):
        dispatcher = RemotePrefillDispatcher()
        target = MembraneNode("target")
        result = dispatcher.dispatch(list(range(512)), "model-a", target)
        assert result.kv_size_mib > 0.0
        assert result.latency_seconds > 0.0
