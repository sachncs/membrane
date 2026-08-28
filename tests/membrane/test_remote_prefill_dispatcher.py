"""Tests for remote_prefill_dispatcher module."""

import pytest

from membrane.adapter import Adapter, PrefillResult
from membrane.fragment import Fragment
from membrane.node import Node
from membrane.prefill_remote import PrefillRemote
from membrane.signature import Signature


class TestRemotePrefillDispatcher:
    """Test suite for PrefillRemote."""

    def test_dispatch_stores_fragments_on_target(self):
        dispatcher = PrefillRemote()
        target = Node("target")
        tokens = list(range(100))
        result = dispatcher.dispatch(tokens, "model-a", target)
        assert isinstance(result, PrefillResult)
        assert len(result.fragments) > 0
        for frag in result.fragments:
            assert target.retrieve(frag.content_hash) is not None

    def test_dispatch_empty_prompt(self):
        dispatcher = PrefillRemote()
        target = Node("target")
        result = dispatcher.dispatch([], "model-a", target)
        assert result.fragments == []

    def test_custom_adapter(self):
        adapter = Adapter(compute_scale=0.5)
        dispatcher = PrefillRemote(prefill_adapter=adapter)
        assert dispatcher.prefill_adapter.compute_scale == 0.5

    def test_dispatch_returns_kvs_estimate(self):
        dispatcher = PrefillRemote()
        target = Node("target")
        result = dispatcher.dispatch(list(range(512)), "model-a", target)
        assert result.kv_size > 0.0
        assert result.latency_seconds > 0.0
