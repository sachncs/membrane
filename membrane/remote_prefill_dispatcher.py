"""RemotePrefillDispatcher: dispatch prefill to optimal remote node."""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillAdapter, PrefillResult


class RemotePrefillDispatcher:
    """Dispatches prefill requests to a chosen remote node.

    In a real system this would be an RPC call. Here it models the
    dispatch by running the prefill on the target node's adapter.
    """

    def __init__(self, prefill_adapter: PrefillAdapter | None = None) -> None:
        """Initialize with optional prefill adapter.

        Args:
            prefill_adapter: Adapter used for remote prefill simulation.
        """
        """Initialize with optional prefill adapter.

        Args:
            prefill_adapter: Adapter used for remote prefill simulation.
        """
        self.prefill_adapter = prefill_adapter or PrefillAdapter()

    def dispatch(
        self,
        prompt_tokens: list[int],
        model_id: str,
        target_node: MembraneNode,
    ) -> PrefillResult:
        """Simulate remote prefill on the target node.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            target_node: Node chosen for prefill.

        Returns:
            PrefillResult with fragments and metadata.
        """
        result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        # Store resulting fragments on the target node
        for frag in result.fragments:
            target_node.store(frag, is_primary=False)
        return result
