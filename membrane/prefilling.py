"""Prefilling pipeline: analytical adapter + dispatch.

This module hosts the prefill pipeline that used to live
across :mod:`membrane.adapter` and :mod:`membrane.prefiller`.
The pipeline is shaped as three layers:

* :class:`Adapter` — wraps the analytical throughput model
  from :mod:`membrane.model.profiler` so callers receive a
  list of fragments plus an optional
  :class:`~membrane.model.router.RoutingDecision`.
* :class:`Prefiller` — schedules one or more prefill
  requests. ``dispatch`` races a list of candidate nodes and
  falls back to local on failure; ``dispatch_sync`` runs
  prefill on a single pre-chosen target.

The pipeline is used by ``Reconstructor`` for prefill
fallback when the index can't fully cover a request. The
production serving plane (``membrane.transport.ops``) calls
``Backend.prefill(...)`` directly without going through
this pipeline.
"""

import asyncio
import logging
from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmenter import Fragmenter
from membrane.model.profiler import kv_size, prefill_time
from membrane.model.router import Router, RoutingDecision
from membrane.node import Node

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefillResult:
    """Outcome of a simulated prefill operation.

    Attributes:
        kv_size: Estimated KV cache size in MiB.
        latency_seconds: Estimated prefill latency in seconds.
        routing_decision: Optional routing decision from
            :class:`Router`. ``None`` when no router is
            configured.
        fragments: Fragments produced from the KV output.
    """

    kv_size: float
    latency_seconds: float
    routing_decision: RoutingDecision | None
    fragments: list[Fragment]


class Adapter:
    """Adapts the Membrane analytical model for integration.

    Treats profiler functions as a model-based prefill service.
    Returns synthetic KV metadata that is immediately converted
    to fragments.

    Attributes:
        router: Optional :class:`Router` consulted for offload
            decisions.
        compute_scale: Hardware compute scale factor.
            ``1.0`` represents the H200 reference hardware.
        fragmentation_engine: Engine used to convert the
            simulated KV output into fragments.
    """

    def __init__(
        self,
        router: Router | None = None,
        compute_scale: float = 1.0,
        fragmentation_engine: Fragmenter | None = None,
    ) -> None:
        """Initialize the adapter."""
        self.router = router
        self.compute_scale = compute_scale
        self.fragmentation_engine = fragmentation_engine or Fragmenter()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        """Simulate prefill and return KV metadata plus fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            PrefillResult: Estimated sizes, latency, optional
            routing decision, and the corresponding fragment
            chain.
        """
        length = len(prompt_tokens)
        size = kv_size(length)
        latency = prefill_time(length, self.compute_scale)

        decision: RoutingDecision | None = None
        if self.router is not None:
            decision = self.router.route(length)

        fragments = self.kv_fragments(prompt_tokens, model_id, size)

        return PrefillResult(
            kv_size=size,
            latency_seconds=latency,
            routing_decision=decision,
            fragments=fragments,
        )

    def kv_fragments(
        self,
        prompt_tokens: list[int],
        model_id: str,
        kv_size: float,
    ) -> list[Fragment]:
        """Convert simulated KV output into content-addressed fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            kv_size: Total KV size to distribute across
                fragments.

        Returns:
            list[Fragment]: Fragments representing the KV tensor
            windows. Empty when ``prompt_tokens`` is empty or
            fragmentation yields no windows.
        """
        if not prompt_tokens:
            return []

        frags = self.fragmentation_engine.create_windows(prompt_tokens, model_id)
        if not frags:
            return []

        total_prompt_tokens = len(prompt_tokens)
        bytes_per_token = (kv_size * 1024.0 * 1024.0) / total_prompt_tokens

        sized_frags: list[Fragment] = []
        for frag in frags:
            span = frag.structural_signature.token_span
            num_tokens = span[1] - span[0] + 1
            frag_size = int(num_tokens * bytes_per_token)

            sized_frags.append(
                Fragment(
                    content_hash=frag.content_hash,
                    embedding=frag.embedding,
                    structural_signature=frag.structural_signature,
                    size=max(1, frag_size),
                    ttl=frag.ttl,
                    reuse_score=frag.reuse_score,
                    version_id=frag.version_id,
                )
            )

        return sized_frags


class PrefillFallbackError(RuntimeError):
    """Raised when all remote prefill attempts fail and no local fallback is available."""


class NodePrefillError(Exception):
    """Raised internally when a single node's prefill attempt fails."""


class Prefiller:
    """Dispatches prefill requests, async across a race or sync to a fixed target.

    The async :meth:`dispatch` races remote nodes and returns the
    first successful result. If all remote nodes fail or time out,
    it falls back to local prefill.

    The sync :meth:`dispatch_sync` runs prefill on a single chosen
    target node and stores the resulting fragments there.

    Attributes:
        prefill_adapter: Adapter that performs the actual
            prefill computation.
        timeout_seconds: Per-node timeout for the async race.
        latency_provider: ``node_id -> latency_seconds`` mapping
            used to simulate network latency in tests.
    """

    def __init__(
        self,
        prefill_adapter: Adapter | None = None,
        timeout_seconds: float = 5.0,
        latency_provider: dict[str, float] | None = None,
    ) -> None:
        """Initialize the dispatcher."""
        self.prefill_adapter = prefill_adapter or Adapter()
        self.timeout_seconds = timeout_seconds
        self.latency_provider = latency_provider or {}

    async def dispatch(
        self,
        prompt_tokens: list[int],
        model_id: str,
        candidate_nodes: list[Node],
        local_node: Node | None = None,
    ) -> PrefillResult:
        """Race remote nodes, fall back to local on failure.

        Schedules one :meth:`try_node` task per candidate, each
        guarded by :func:`asyncio.wait_for`. As tasks complete
        the first successful result is returned and the
        remaining tasks are cancelled.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            candidate_nodes: Nodes to attempt prefill on.
            local_node: Optional local node for fallback.

        Returns:
            PrefillResult: From the first successful remote
            attempt, or from the local fallback.

        Raises:
            PrefillFallbackError: When no remote candidate
            succeeds and no local fallback is available.
        """
        if self.prefill_adapter.router is not None:
            try:
                decision_result = self.prefill_adapter.prefill(prompt_tokens, model_id)
                if (
                    decision_result.routing_decision is not None
                    and decision_result.routing_decision.target == "pd-p"
                    and local_node is not None
                ):
                    for frag in decision_result.fragments:
                        local_node.store(frag, is_primary=True)
                    return decision_result
            except Exception as exc:
                logger.debug("analytical prefill for routing decision failed: %s", exc)

        if not candidate_nodes:
            if local_node is None:
                raise PrefillFallbackError("No candidate nodes and no local fallback")
            return self.local_prefill(prompt_tokens, model_id, local_node)

        timeout_tasks = [
            asyncio.create_task(
                asyncio.wait_for(
                    self.try_node(prompt_tokens, model_id, node),
                    timeout=self.timeout_seconds,
                )
            )
            for node in candidate_nodes
        ]

        try:
            for coro in asyncio.as_completed(timeout_tasks):
                try:
                    result = await coro
                    for t in timeout_tasks:
                        t.cancel()
                    return result
                except (asyncio.TimeoutError, NodePrefillError):
                    continue
        finally:
            for t in timeout_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*timeout_tasks, return_exceptions=True)

        if local_node is not None:
            logger.info("All remote prefill attempts failed; falling back to local")
            return self.local_prefill(prompt_tokens, model_id, local_node)
        raise PrefillFallbackError("All remote prefill attempts failed")

    async def try_node(
        self,
        prompt_tokens: list[int],
        model_id: str,
        node: Node,
    ) -> PrefillResult:
        """Attempt prefill on a single node, simulating network latency.

        Raises:
            NodePrefillError: When the underlying adapter
                raises, or when it returns no fragments.
        """
        latency = self.latency_provider.get(node.node_id, 0.0)
        if latency > 0:
            await asyncio.sleep(latency)

        try:
            result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        except Exception as exc:
            raise NodePrefillError(f"Node {node.node_id} prefill failed: {exc}") from exc
        if not result.fragments:
            raise NodePrefillError(f"Node {node.node_id} returned empty fragments")
        for frag in result.fragments:
            node.store(frag, is_primary=False)
        return result

    def local_prefill(
        self,
        prompt_tokens: list[int],
        model_id: str,
        local_node: Node,
    ) -> PrefillResult:
        """Run prefill locally and store fragments as primary."""
        result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        for frag in result.fragments:
            local_node.store(frag, is_primary=True)
        return result

    def dispatch_sync(
        self,
        prompt_tokens: list[int],
        model_id: str,
        target_node: Node,
    ) -> PrefillResult:
        """Run prefill on a single pre-chosen target node.

        Synchronous counterpart to :meth:`dispatch`. The target
        node is chosen by the caller (this method does not race);
        the resulting fragments are stored on the target as
        non-primary replicas.

        Raises:
            NodePrefillError: When the adapter raises or returns
                no fragments.
        """
        try:
            result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        except Exception as exc:
            raise NodePrefillError(f"Node {target_node.node_id} prefill failed: {exc}") from exc
        if not result.fragments:
            raise NodePrefillError(f"Node {target_node.node_id} returned empty fragments")
        for frag in result.fragments:
            target_node.store(frag, is_primary=False)
        return result


__all__ = [
    "Adapter",
    "NodePrefillError",
    "PrefillFallbackError",
    "PrefillResult",
    "Prefiller",
]
