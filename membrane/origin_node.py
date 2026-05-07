"""OriginNode: canonical memory authority that propagates to replicas."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService


class OriginNode(MembraneNode):
    """Canonical memory authority in a regional topology.

    Receives all primary writes and can propagate fragments to replicas.
    """

    def __init__(
        self,
        node_id: str,
        max_memory_bytes: int = 1 << 30,
        transfer_service: TransferService | None = None,
    ) -> None:
        """Initialize the origin node.

        Args:
            node_id: Unique identifier.
            max_memory_bytes: Memory budget in bytes.
            transfer_service: Optional transfer service for replica pushes.
        """
        super().__init__(node_id, max_memory_bytes)
        self.transfer_service = transfer_service or TransferService()

    def promote_to_replica(
        self,
        fragment: Fragment,
        replica: MembraneNode,
    ) -> bool:
        """Push a fragment to a replica node.

        Args:
            fragment: Fragment to propagate.
            replica: Target replica node.

        Returns:
            True if transfer succeeded.
        """
        if fragment.content_hash not in self.fragments:
            self.store(fragment, is_primary=True)
        return self.transfer_service.transfer_fragment(
            self, replica, fragment.content_hash
        )

    def bulk_promote(
        self,
        content_hashes: list[str],
        replica: MembraneNode,
    ) -> list[str]:
        """Push multiple fragments to a replica.

        Args:
            content_hashes: Hashes to propagate.
            replica: Target replica node.

        Returns:
            List of successfully transferred hashes.
        """
        transferred: list[str] = []
        for h in content_hashes:
            if self.transfer_service.transfer_fragment(self, replica, h):
                transferred.append(h)
        return transferred
