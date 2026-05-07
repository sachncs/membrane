"""ReplicaNode: hot regional cache with warming support."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.origin_node import OriginNode
from membrane.transfer_service import TransferService


class ReplicaNode(MembraneNode):
    """Hot regional cache that holds non-primary copies of fragments.

    Supports warming from an origin node on demand or proactively.
    """

    def __init__(
        self,
        node_id: str,
        max_memory_bytes: int = 1 << 30,
        transfer_service: TransferService | None = None,
    ) -> None:
        """Initialize the replica node.

        Args:
            node_id: Unique identifier.
            max_memory_bytes: Memory budget in bytes.
            transfer_service: Optional transfer service for origin fetches.
        """
        super().__init__(node_id, max_memory_bytes)
        self.transfer_service = transfer_service or TransferService()

    def warm_from_origin(
        self,
        origin: OriginNode,
        content_hashes: list[str],
    ) -> list[str]:
        """Bulk-fetch fragments from the origin to warm the cache.

        Args:
            origin: Origin node to fetch from.
            content_hashes: Hashes to warm.

        Returns:
            List of successfully warmed hashes.
        """
        warmed: list[str] = []
        for h in content_hashes:
            if self.transfer_service.transfer_fragment(origin, self, h):
                warmed.append(h)
        return warmed

    def store(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Store a fragment. Replicas never own primary shards.

        Args:
            fragment: Fragment to store.
            is_primary: Ignored; always stored as non-primary.

        Returns:
            True if stored.
        """
        return super().store(fragment, is_primary=False)
