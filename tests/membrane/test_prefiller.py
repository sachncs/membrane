"""Tests for the synchronous dispatch_sync path on Prefill."""

import pytest

from membrane.node import Node
from membrane.prefilling import Adapter, PrefillResult
from membrane.prefilling import Prefiller as PrefillRemote


class TestPrefillerSync:
    """Test suite for Prefiller.dispatch_sync."""

    def test_dispatch_sync_stores_fragments_on_target(self):
        dispatcher = PrefillRemote()
        target = Node("target")
        tokens = list(range(100))
        result = dispatcher.dispatch_sync(tokens, "model-a", target)
        assert isinstance(result, PrefillResult)
        assert len(result.fragments) > 0
        for frag in result.fragments:
            assert target.retrieve(frag.identity.payload_hash) is not None

    def test_dispatch_sync_empty_prompt(self):
        """An empty prompt produces no fragments; dispatch_sync
        surfaces the empty result via NodePrefillError so callers
        can distinguish 'nothing to ship' from 'succeeded with
        fragments' (consistent with the async try_node path)."""
        from membrane.prefilling import NodePrefillError

        dispatcher = PrefillRemote()
        target = Node("target")
        with pytest.raises(NodePrefillError):
            dispatcher.dispatch_sync([], "model-a", target)

    def test_custom_adapter(self):
        adapter = Adapter(compute_scale=0.5)
        dispatcher = PrefillRemote(prefill_adapter=adapter)
        assert dispatcher.prefill_adapter.compute_scale == 0.5

    def test_dispatch_sync_returns_kvs_estimate(self):
        dispatcher = PrefillRemote()
        target = Node("target")
        result = dispatcher.dispatch_sync(list(range(512)), "model-a", target)
        assert result.kv_size > 0.0
        assert result.latency_seconds > 0.0
