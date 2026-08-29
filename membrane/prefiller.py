"""Prefiller: concurrent + synchronous prefill dispatch.

This module defines :class:`Prefiller`, the unified prefill
dispatcher. Two modes are exposed:

* :meth:`Prefiller.dispatch` — async; races prefill requests across
  multiple candidate nodes and returns the first successful
  result. Falls back to local prefill (if a local node is
  supplied) when every remote attempt fails or times out.
* :meth:`Prefiller.dispatch_sync` — sync; runs prefill on a single
  pre-chosen target node. This is the synchronous counterpart
  used when the dispatcher has already chosen a specific node
  (e.g., for deterministic tests and for the Membrane transport
  prefill route).

The module also defines two exception types used by the
dispatcher:

* :class:`PrefillFallbackError` — raised when no remote
  succeeds and no local fallback is available.
* :class:`NodePrefillError` — raised internally when a single
  node fails or returns an empty fragment.

Network latency is simulated via a configurable
``latency_provider`` dict; in a real system this would be the
network RTT measurement layer.
"""

import asyncio
import logging

from membrane.adapter import Adapter, PrefillResult
from membrane.node import Node

logger = logging.getLogger(__name__)


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

    In a real system the latency would be network RTT; here it
    is configurable via ``latency_provider`` for simulation.

    Attributes:
        prefill_adapter: Adapter that performs the actual
            prefill computation (CPU/GPU/Transformers/etc.).
        timeout_seconds: Per-node timeout for the async race. A
            node that does not respond in this window is treated
            as failed.
        latency_provider: Optional ``node_id -> latency_seconds``
            mapping used to simulate network latency in tests.
    """

    def __init__(
        self,
        prefill_adapter: Adapter | None = None,
        timeout_seconds: float = 5.0,
        latency_provider: dict[str, float] | None = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            prefill_adapter: Adapter used for prefill simulation.
                A default :class:`Adapter` is created
                when ``None``.
            timeout_seconds: Max seconds to wait for each
                remote node.
            latency_provider: Mapping
                ``node_id -> latency_seconds``.
        """
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

        Schedules one ``try_node`` task per candidate, each
        guarded by :func:`asyncio.wait_for`. As tasks complete
        the first successful result is returned and the
        remaining tasks are cancelled. If every remote attempt
        fails or times out, the dispatcher falls back to
        :meth:`local_prefill` when a ``local_node`` is provided.

        The dispatcher consults the
        :class:`~membrane.adapter.Adapter` to produce a
        :class:`~membrane.model.router.RoutingDecision` before
        racing. A ``target="pd-p"`` decision skips the remote
        race entirely and goes straight to the local node (if
        provided); a ``target="membrane"`` decision races the
        candidates as usual.

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
        # Honor the analytical routing decision when the adapter
        # has a router configured. A target="pd-p" decision means
        # the analytical model judged local compute sufficient; we
        # skip the remote race and serve locally. The check is
        # gated on a configured router so adapters that only
        # produce fragments (and don't return a routing decision)
        # are not perturbed.
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
                # Decision absent, target != 'pd-p', or no local
                # node: fall through to the race.
            except Exception as exc:  # noqa: BLE001
                logger.debug("analytical prefill for routing decision failed: %s", exc)

        if not candidate_nodes:
            if local_node is None:
                raise PrefillFallbackError("No candidate nodes and no local fallback")
            return self.local_prefill(prompt_tokens, model_id, local_node)

        # Schedule a guarded task per candidate.
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
            # Walk the results as they complete; return on the
            # first success, swallow timeouts and per-node
            # errors, and keep iterating.
            for coro in asyncio.as_completed(timeout_tasks):
                try:
                    result = await coro
                    for t in timeout_tasks:
                        t.cancel()
                    return result
                except (asyncio.TimeoutError, NodePrefillError):
                    continue
        finally:
            # Ensure all tasks are cleaned up even if the
            # caller cancels us mid-flight.
            for t in timeout_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*timeout_tasks, return_exceptions=True)

        # All remotes failed or timed out.
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

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            node: Target node.

        Returns:
            PrefillResult: The prefill result, after storing
            each fragment on the target node as a non-primary
            replica.

        Raises:
            NodePrefillError: When the underlying adapter
            raises, or when it returns no fragments.
        """
        latency = self.latency_provider.get(node.node_id, 0.0)
        if latency > 0:
            # Simulate network latency in tests. In production
            # this is replaced by real RPC awaiting.
            await asyncio.sleep(latency)

        try:
            result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        except Exception as exc:
            raise NodePrefillError(f"Node {node.node_id} prefill failed: {exc}") from exc
        if not result.fragments:
            raise NodePrefillError(f"Node {node.node_id} returned empty fragments")
        # Store every fragment on the target as a non-primary
        # replica so subsequent reads can find it there.
        for frag in result.fragments:
            node.store(frag, is_primary=False)
        return result

    def local_prefill(
        self,
        prompt_tokens: list[int],
        model_id: str,
        local_node: Node,
    ) -> PrefillResult:
        """Run prefill locally and store fragments as primary.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            local_node: Local node to host the fragments.

        Returns:
            PrefillResult: The prefill result, after storing
            each fragment on the local node as a primary
            owner.
        """
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

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            target_node: Node chosen for prefill.

        Returns:
            PrefillResult: Fragments and prefill metadata after
            each fragment is stored on the target as a
            non-primary replica.

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


__all__ = ["PrefillFallbackError", "NodePrefillError", "Prefiller"]
