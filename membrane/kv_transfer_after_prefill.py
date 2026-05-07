"""KVTransferAfterPrefill: ship KV fragments back to requesting node after remote prefill."""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillResult
from membrane.transfer_service import TransferService


class KVTransferAfterPrefill:
    """Transfers KV fragments from the compute node back to the requester.

    Uses TransferService to move all fragments produced by a remote prefill.
    """

    def __init__(self, transfer_service: TransferService | None = None) -> None:
        """Initialize with optional transfer service.

        Args:
            transfer_service: Service used for fragment movement.
        """
        """Initialize with optional transfer service.

        Args:
            transfer_service: Service used for fragment movement.
        """
        self.transfer_service = transfer_service or TransferService()

    def ship_kv(
        self,
        prefill_result: PrefillResult,
        source_node: MembraneNode,
        target_node: MembraneNode,
    ) -> list[str]:
        """Ship all fragments from prefill result to the target node.

        Args:
            prefill_result: Result containing fragments to transfer.
            source_node: Node that performed the prefill.
            target_node: Node requesting the KV.

        Returns:
            List of successfully transferred content hashes.
        """
        transferred: list[str] = []
        for frag in prefill_result.fragments:
            if self.transfer_service.transfer_fragment(
                source_node, target_node, frag.content_hash
            ):
                transferred.append(frag.content_hash)
        return transferred
