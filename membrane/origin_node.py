"""OriginNode: canonical memory authority that propagates to replicas.

This module defines :class:`OriginNode`, a specialized
:class:`~membrane.membrane_node.MembraneNode` that acts as the
canonical authority for a region. In addition to the regular
in-memory store, an origin can push fragments to replica nodes
via a :class:`~membrane.transfer_service.TransferService`.

Writes flow through the origin so that a single source of truth
exists for each fragment; reads can be served by either the origin
or any of its replicas. Promotion to a replica is explicit and
idempotent — re-promoting an already-present fragment is a no-op
on the wire.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService


class OriginNode(MembraneNode):
    """Canonical memory authority in a regional topology.

    Receives all primary writes and can propagate fragments to
    replicas.

    Attributes:
        transfer_service: Service used to push fragments to
            replicas. Held by reference so callers can substitute
            a custom implementation (e.g., one that records
            transfers for testing).
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
            transfer_service: Optional transfer service for
                replica pushes. A default
                :class:`TransferService` is created when ``None``.
        """
        super().__init__(node_id, max_memory_bytes)
        self.transfer_service = transfer_service or TransferService()

    def promote_to_replica(
        self,
        fragment: Fragment,
        replica: MembraneNode,
    ) -> bool:
        """Push a fragment to a replica node.

        If the origin does not already hold the fragment, it is
        stored as a primary before the transfer is attempted.

        Args:
            fragment: Fragment to propagate.
            replica: Target replica node.

        Returns:
            bool: True if the transfer succeeded, False
            otherwise.
        """
        if fragment.content_hash not in self.fragments:
            # Ensure the origin is the source of truth for the
            # fragment before pushing it downstream.
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

        Iterates over ``content_hashes`` and attempts each
        transfer independently. A failed transfer is logged but
        does not abort the rest of the batch.

        Args:
            content_hashes: Hashes to propagate.
            replica: Target replica node.

        Returns:
            list[str]: Content hashes whose transfer succeeded.
        """
        transferred: list[str] = []
        for h in content_hashes:
            if self.transfer_service.transfer_fragment(self, replica, h):
                transferred.append(h)
        return transferred
