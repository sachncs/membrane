"""Tests for AsyncRemotePrefillDispatcher designed to break the algorithm."""

import asyncio

import pytest

from membrane.async_prefill_dispatcher import (
    AsyncRemotePrefillDispatcher,
    PrefillFallbackError,
)
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillAdapter, PrefillResult
from membrane.structural_signature import StructuralSignature


class EmptyFragmentAdapter(PrefillAdapter):
    """Adapter that always returns empty fragments."""

    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        return PrefillResult(
            kv_size_mib=0.0,
            latency_seconds=0.0,
            routing_decision=None,
            fragments=[],
        )


class ExplodingAdapter(PrefillAdapter):
    """Adapter that always raises."""

    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        raise RuntimeError("simulated prefill failure")


class ControlledAdapter(PrefillAdapter):
    """Adapter that returns a predictable fragment."""

    def __init__(self, hash_prefix: str = "ctrl") -> None:
        super().__init__()
        self.hash_prefix = hash_prefix
        self.call_count = 0

    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        self.call_count += 1
        length = len(prompt_tokens)
        frag = Fragment(
            content_hash=f"{self.hash_prefix}-{length}",
            embedding=(0.1,),
            structural_signature=StructuralSignature(
                model_id=model_id, layer_range=(0, 1), token_span=(0, length - 1)
            ),
            size=10,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )
        return PrefillResult(
            kv_size_mib=1.0,
            latency_seconds=0.01,
            routing_decision=None,
            fragments=[frag],
        )


@pytest.mark.anyio
async def test_fastest_node_wins_race():
    """When nodes have different latencies, the fastest should win."""
    adapter = ControlledAdapter("fast")
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=1.0,
        latency_provider={"fast": 0.01, "medium": 0.05, "slow": 0.10},
    )

    fast = MembraneNode("fast")
    medium = MembraneNode("medium")
    slow = MembraneNode("slow")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [fast, medium, slow], local_node=None
    )

    # Fastest node should have stored the fragment
    assert fast.retrieve(f"fast-10") is not None
    # Slower nodes may or may not have stored depending on cancellation timing,
    # but the *returned* result must come from the winning adapter call.
    assert result.kv_size_mib == 1.0
    assert result.fragments[0].content_hash == "fast-10"


@pytest.mark.anyio
async def test_timeout_cancels_slow_node():
    """A node slower than timeout must not win and should be cancelled."""
    adapter = ControlledAdapter("win")
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=0.03,
        latency_provider={"win": 0.01, "lose": 0.10},
    )

    win = MembraneNode("win")
    lose = MembraneNode("lose")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [win, lose], local_node=None
    )

    assert result.fragments[0].content_hash == "win-10"
    assert win.retrieve("win-10") is not None
    # Slow node task is cancelled before store completes
    assert lose.retrieve("win-10") is None


@pytest.mark.anyio
async def test_all_timeout_fallback_local():
    """If every remote node times out, fall back to local prefill."""
    adapter = ControlledAdapter("local")
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=0.01,
        latency_provider={"slow1": 0.10, "slow2": 0.10},
    )

    local = MembraneNode("local")
    slow1 = MembraneNode("slow1")
    slow2 = MembraneNode("slow2")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [slow1, slow2], local_node=local
    )

    assert result.fragments[0].content_hash == "local-10"
    assert local.retrieve("local-10") is not None
    assert "local-10" in local.get_shard_hashes()


@pytest.mark.anyio
async def test_all_timeout_no_local_raises():
    """If all remotes time out and no local node is given, raise."""
    adapter = ControlledAdapter("never")
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=0.01,
        latency_provider={"slow": 0.10},
    )

    with pytest.raises(PrefillFallbackError):
        await dispatcher.dispatch(
            list(range(10)), "m", [MembraneNode("slow")], local_node=None
        )


@pytest.mark.anyio
async def test_empty_fragments_treated_as_failure():
    """A node returning empty fragments should be treated as a failed attempt."""
    bad = EmptyFragmentAdapter()
    good = ControlledAdapter("good")

    # Use a composite adapter that alternates? No, we need two *different*
    # adapters on two nodes.  The dispatcher uses a single adapter for all
    # nodes, so we simulate this by having the adapter behave differently
    # per call using a counter.
    class AlternatingAdapter(PrefillAdapter):
        def __init__(self) -> None:
            self.calls = 0

        def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
            self.calls += 1
            if self.calls == 1:
                return PrefillResult(
                    kv_size_mib=0.0,
                    latency_seconds=0.0,
                    routing_decision=None,
                    fragments=[],
                )
            return good.prefill(prompt_tokens, model_id)

    adapter = AlternatingAdapter()
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=1.0,
        latency_provider={"bad": 0.01, "good": 0.02},
    )

    bad_node = MembraneNode("bad")
    good_node = MembraneNode("good")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [bad_node, good_node], local_node=None
    )

    assert result.fragments[0].content_hash == "good-10"
    assert good_node.retrieve("good-10") is not None


@pytest.mark.anyio
async def test_exception_in_prefill_treated_as_failure():
    """A node whose adapter raises should not crash the dispatcher."""
    bad = ExplodingAdapter()
    good = ControlledAdapter("good")

    class AlternatingExplodingAdapter(PrefillAdapter):
        def __init__(self) -> None:
            self.calls = 0

        def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return good.prefill(prompt_tokens, model_id)

    adapter = AlternatingExplodingAdapter()
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=1.0,
        latency_provider={"bad": 0.01, "good": 0.02},
    )

    bad_node = MembraneNode("bad")
    good_node = MembraneNode("good")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [bad_node, good_node], local_node=None
    )

    assert result.fragments[0].content_hash == "good-10"


@pytest.mark.anyio
async def test_no_candidates_uses_local():
    """Empty candidate list with a local node should use local prefill."""
    adapter = ControlledAdapter("local")
    dispatcher = AsyncRemotePrefillDispatcher(prefill_adapter=adapter)

    local = MembraneNode("local")
    result = await dispatcher.dispatch(
        list(range(10)), "m", [], local_node=local
    )

    assert result.fragments[0].content_hash == "local-10"
    assert local.retrieve("local-10") is not None
    assert "local-10" in local.get_shard_hashes()


@pytest.mark.anyio
async def test_no_candidates_no_local_raises():
    """Empty candidate list with no local node should raise."""
    dispatcher = AsyncRemotePrefillDispatcher()
    with pytest.raises(PrefillFallbackError):
        await dispatcher.dispatch(list(range(10)), "m", [], local_node=None)


@pytest.mark.anyio
async def test_partial_failure_one_succeeds():
    """One node fails, another succeeds; success should be returned."""
    class OneShotFailAdapter(PrefillAdapter):
        def __init__(self) -> None:
            self.fail_node_id: str | None = None

        def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
            # This adapter is shared, so we can't know which node is calling.
            # Use a trick: the first call fails, second succeeds.
            if not hasattr(self, "calls"):
                self.calls = 0
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first call fails")
            return ControlledAdapter("ok").prefill(prompt_tokens, model_id)

    adapter = OneShotFailAdapter()
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=1.0,
        latency_provider={"fail": 0.01, "ok": 0.02},
    )

    fail_node = MembraneNode("fail")
    ok_node = MembraneNode("ok")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [fail_node, ok_node], local_node=None
    )

    assert result.fragments[0].content_hash == "ok-10"
    assert ok_node.retrieve("ok-10") is not None


@pytest.mark.anyio
async def test_cancellation_cleanup_on_success():
    """Pending tasks must be cancelled and awaited without leaking exceptions."""
    adapter = ControlledAdapter("win")
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=1.0,
        latency_provider={"win": 0.01, "pending1": 0.30, "pending2": 0.30},
    )

    win = MembraneNode("win")
    pending1 = MembraneNode("pending1")
    pending2 = MembraneNode("pending2")

    # This should complete quickly and not hang waiting for pending nodes.
    result = await dispatcher.dispatch(
        list(range(10)), "m", [win, pending1, pending2], local_node=None
    )

    assert result.fragments[0].content_hash == "win-10"
    # After a tiny yield to let cancellations propagate, no pending tasks remain.
    await asyncio.sleep(0.05)
    assert pending1.retrieve("win-10") is None
    assert pending2.retrieve("win-10") is None


@pytest.mark.anyio
async def test_timeout_per_node_not_global():
    """Each node has its own timeout; a global timeout must not starve others."""
    adapter = ControlledAdapter("ok")
    dispatcher = AsyncRemotePrefillDispatcher(
        prefill_adapter=adapter,
        timeout_seconds=0.02,
        latency_provider={"slow": 0.10, "ok": 0.01},
    )

    slow = MembraneNode("slow")
    ok = MembraneNode("ok")

    result = await dispatcher.dispatch(
        list(range(10)), "m", [slow, ok], local_node=None
    )

    assert result.fragments[0].content_hash == "ok-10"
    assert ok.retrieve("ok-10") is not None
