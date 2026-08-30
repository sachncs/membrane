"""Tests for the routing-decision short-circuit in PrefillAsync."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from membrane.fragment import Fragment
from membrane.fragmenter import compute_content_hash
from membrane.model.router import Router, RoutingDecision
from membrane.node import Node
from membrane.prefilling import Adapter, PrefillResult
from membrane.prefilling import Prefiller as PrefillAsync
from membrane.signature import Signature


class _RouterAdapter(Adapter):
    """Adapter that always emits a target='pd-p' routing decision."""

    def __init__(self) -> None:
        super().__init__(router=Router(threshold=1))

    def prefill(self, prompt_tokens, model_id) -> PrefillResult:
        tokens = tuple(prompt_tokens[:1024])
        h = compute_content_hash(tokens)
        frag = Fragment(
            content_hash=h,
            embedding=(float(len(prompt_tokens)),),
            structural_signature=Signature(
                model_id=model_id, layer_range=(0, 1), token_span=(0, len(prompt_tokens) - 1)
            ),
            size=10,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )
        return PrefillResult(
            kv_size=1.0,
            latency_seconds=0.01,
            routing_decision=RoutingDecision(
                target="pd-p",
                incremental_length=len(prompt_tokens),
                cached_prefix_length=0,
            ),
            fragments=[frag],
        )


@pytest.mark.anyio
async def test_pd_p_target_serves_locally_and_skips_remotes():
    """A target='pd-p' decision from the analytical router should send
    the prompt to the local node and never touch remote candidates.
    """
    local = Node("local")
    remote = MagicMock(spec=Node)
    remote.node_id = "remote"
    # If the dispatcher calls .store / .retrieve on the remote, the test
    # will explode, which is exactly the assertion we want.
    adapter = _RouterAdapter()
    dispatcher = PrefillAsync(prefill_adapter=adapter)

    result = await dispatcher.dispatch(list(range(10)), "m", [remote], local_node=local)

    # The decision was honored: only the local node was touched.
    assert local.retrieve(result.fragments[0].content_hash) is not None
    assert local.get_shard_hashes()  # stored as primary
    remote.retrieve.assert_not_called()
    remote.store.assert_not_called()


@pytest.mark.anyio
async def test_membrane_target_falls_through_to_normal_race():
    """A target='membrane' decision must not short-circuit the remote
    race; the dispatcher should still race the candidate nodes.
    """

    class MembraneTargetAdapter(Adapter):
        def __init__(self) -> None:
            super().__init__(router=Router(threshold=1))

        def prefill(self, prompt_tokens, model_id) -> PrefillResult:
            tokens = tuple(prompt_tokens[:1024])
            h = compute_content_hash(tokens)
            return PrefillResult(
                kv_size=1.0,
                latency_seconds=0.0,
                routing_decision=RoutingDecision(
                    target="membrane",
                    incremental_length=len(prompt_tokens),
                    cached_prefix_length=0,
                ),
                fragments=[
                    Fragment(
                        content_hash=h,
                        embedding=(float(len(prompt_tokens)),),
                        structural_signature=Signature(
                            model_id=model_id,
                            layer_range=(0, 1),
                            token_span=(0, len(prompt_tokens) - 1),
                        ),
                        size=10,
                        ttl=3600.0,
                        reuse_score=0.5,
                        version_id=1,
                    )
                ],
            )

    n1 = Node("n1")
    n2 = Node("n2")
    dispatcher = PrefillAsync(prefill_adapter=MembraneTargetAdapter())

    result = await dispatcher.dispatch(list(range(10)), "m", [n1, n2], local_node=None)

    assert result.fragments
    # Both nodes tried; at least one should have stored the fragment.
    assert (
        n1.retrieve(result.fragments[0].content_hash) is not None
        or n2.retrieve(result.fragments[0].content_hash) is not None
    )
