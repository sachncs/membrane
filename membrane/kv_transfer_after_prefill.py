"""KVTransferAfterPrefill: ship KV fragments back to the requester after remote prefill.

This module defines :class:`KVTransferAfterPrefill`, a small
helper that copies every fragment produced by a remote prefill
from the compute node back to the requesting node via the
:class:`~membrane.transfer_service.TransferService`.

The class is intentionally minimal — it is essentially a
loop over the prefill fragments. Keeping it as a dedicated class
gives callers a clear name to depend on and allows tests to
substitute a mock transfer service.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillResult
from membrane.transfer_service import TransferService


class KVTransferAfterPrefill:
    """Transfers KV fragments from the compute node back to the requester.

    Uses :class:`TransferService` to move all fragments produced
    by a remote prefill.

    Attributes:
        transfer_service: Service used for fragment movement.
            Held by reference so callers can substitute a custom
            implementation (e.g., one that records transfers for
            testing).
    """

    def __init__(self, transfer_service: TransferService | None = None) -> None:
        """Initialize with an optional transfer service.

        Args:
            transfer_service: Service used for fragment
                movement. A default :class:`TransferService` is
                created when ``None``.
        """
        self.transfer_service = transfer_service or TransferService()

    def ship_kv(
        self,
        prefill_result: PrefillResult,
        source_node: MembraneNode,
        target_node: MembraneNode,
    ) -> list[str]:
        """Ship every fragment from ``prefill_result`` to ``target_node``.

        Each fragment is transferred independently. Failures are
        silently skipped; the returned list contains only the
        successfully transferred hashes.

        Args:
            prefill_result: Result containing fragments to
                transfer.
            source_node: Node that performed the prefill.
            target_node: Node requesting the KV.

        Returns:
            list[str]: Successfully transferred content hashes.
        """
        transferred: list[str] = []
        for frag in prefill_result.fragments:
            if self.transfer_service.transfer_fragment(source_node, target_node, frag.content_hash):
                transferred.append(frag.content_hash)
        return transferred
