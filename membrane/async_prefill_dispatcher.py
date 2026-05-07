"""AsyncRemotePrefillDispatcher: concurrent prefill dispatch with timeout and fallback."""

import asyncio
import logging

from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillAdapter, PrefillResult


logger = logging.getLogger(__name__)


class PrefillFallbackError(RuntimeError):
    """Raised when all remote prefill attempts fail and no local fallback is available."""


class NodePrefillError(Exception):
    """Exception for a single node prefill failure."""


class AsyncRemotePrefillDispatcher:
    """Dispatches prefill requests concurrently to multiple candidate nodes.

    Races remote nodes and returns the first successful result.
    If all remote nodes fail or time out, falls back to local prefill.

    In a real system the latency would be network RTT; here it is
    configurable via *latency_provider* for simulation.
    """

    def __init__(
        self,
        prefill_adapter: PrefillAdapter | None = None,
        timeout_seconds: float = 5.0,
        latency_provider: dict[str, float] | None = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            prefill_adapter: Adapter used for prefill simulation.
            timeout_seconds: Max seconds to wait for each remote node.
            latency_provider: Mapping ``node_id -> latency_seconds``.
        """
        self.prefill_adapter = prefill_adapter or PrefillAdapter()
        self.timeout_seconds = timeout_seconds
        self.latency_provider = latency_provider or {}

    async def dispatch(
        self,
        prompt_tokens: list[int],
        model_id: str,
        candidate_nodes: list[MembraneNode],
        local_node: MembraneNode | None = None,
    ) -> PrefillResult:
        """Race remote nodes, fallback to local on failure.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            candidate_nodes: Nodes to attempt prefill on.
            local_node: Optional local node for fallback.

        Returns:
            PrefillResult from the first successful remote or local fallback.

        Raises:
            PrefillFallbackError: If no remote succeeds and no local fallback.
        """
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
            # Ensure all tasks are cleaned up even if the caller cancels us.
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
        node: MembraneNode,
    ) -> PrefillResult:
        """Attempt prefill on a single node, simulating network latency."""
        latency = self.latency_provider.get(node.node_id, 0.0)
        if latency > 0:
            await asyncio.sleep(latency)

        try:
            result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        except Exception as exc:
            raise NodePrefillError(
                f"Node {node.node_id} prefill failed: {exc}"
            ) from exc
        if not result.fragments:
            raise NodePrefillError(
                f"Node {node.node_id} returned empty fragments"
            )
        for frag in result.fragments:
            node.store(frag, is_primary=False)
        return result

    def local_prefill(
        self,
        prompt_tokens: list[int],
        model_id: str,
        local_node: MembraneNode,
    ) -> PrefillResult:
        """Run prefill locally and store fragments as primary."""
        result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        for frag in result.fragments:
            local_node.store(frag, is_primary=True)
        return result
