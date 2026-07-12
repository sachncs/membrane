"""RemotePrefillDispatcher: dispatch prefill to a single chosen remote node.

This module defines :class:`RemotePrefillDispatcher`, a
synchronous counterpart to
:class:`~membrane.async_prefill_dispatcher.AsyncRemotePrefillDispatcher`.
It runs prefill on a single pre-chosen remote node, stores the
resulting fragments on that node as non-primary replicas, and
returns the :class:`~membrane.prefill_adapter.PrefillResult`.

The dispatcher models a remote RPC call with a local simulation
— the ``target_node`` argument is used to host the resulting
fragments, but no real network I/O occurs. Use
:class:`AsyncRemotePrefillDispatcher` when concurrent racing is
required.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillAdapter, PrefillResult


class RemotePrefillDispatcher:
    """Dispatches prefill requests to a chosen remote node.

    In a real system this would be an RPC call. Here it models
    the dispatch by running the prefill on the target node's
    adapter.

    Attributes:
        prefill_adapter: Adapter that performs the actual
            prefill computation.
    """

    def __init__(self, prefill_adapter: PrefillAdapter | None = None) -> None:
        """Initialize with an optional prefill adapter.

        Args:
            prefill_adapter: Adapter used for remote prefill
                simulation. A default :class:`PrefillAdapter`
                is created when ``None``.
        """
        self.prefill_adapter = prefill_adapter or PrefillAdapter()

    def dispatch(
        self,
        prompt_tokens: list[int],
        model_id: str,
        target_node: MembraneNode,
    ) -> PrefillResult:
        """Simulate remote prefill on ``target_node``.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            target_node: Node chosen for prefill.

        Returns:
            PrefillResult: Fragments and prefill metadata after
            each fragment is stored on the target as a
            non-primary replica.
        """
        result = self.prefill_adapter.prefill(prompt_tokens, model_id)
        # Store resulting fragments on the target node.
        for frag in result.fragments:
            target_node.store(frag, is_primary=False)
        return result
