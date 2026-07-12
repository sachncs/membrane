"""ReplicaNode: hot regional cache with warming support.

This module defines :class:`ReplicaNode`, a specialized
:class:`~membrane.membrane_node.MembraneNode` that holds
*non-primary* copies of fragments. Replicas exist to reduce read
latency for nearby clients and to absorb traffic from the origin.

Two mechanisms are exposed:

* :meth:`warm_from_origin` — proactively pull a batch of hashes
  from an :class:`~membrane.origin_node.OriginNode` so that the
  replica is ready to serve them when requests arrive.
* :meth:`store` — overridden to force ``is_primary=False``,
  preventing the replica from accidentally claiming primary
  ownership of a fragment.

The replica delegates transport to a
:class:`~membrane.transfer_service.TransferService` so the same
transport used for gossip can also be used for warming.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.origin_node import OriginNode
from membrane.transfer_service import TransferService


class ReplicaNode(MembraneNode):
    """Hot regional cache that holds non-primary copies of fragments.

    Supports warming from an origin node on demand or
    proactively.

    Attributes:
        transfer_service: Service used to fetch fragments from
            an origin. Held by reference so callers can inject a
            custom implementation for tests.
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
            transfer_service: Optional transfer service for
                origin fetches. A default
                :class:`TransferService` is created when ``None``.
        """
        super().__init__(node_id, max_memory_bytes)
        self.transfer_service = transfer_service or TransferService()

    def warm_from_origin(
        self,
        origin: OriginNode,
        content_hashes: list[str],
    ) -> list[str]:
        """Bulk-fetch fragments from the origin to warm the cache.

        Each hash is fetched independently. Failures are skipped
        silently and recorded as missing from the returned list.

        Args:
            origin: Origin node to fetch from.
            content_hashes: Hashes to warm.

        Returns:
            list[str]: Content hashes that were successfully
            transferred to the replica.
        """
        warmed: list[str] = []
        for h in content_hashes:
            if self.transfer_service.transfer_fragment(origin, self, h):
                warmed.append(h)
        return warmed

    def store(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Store a fragment. Replicas never own primary shards.

        The ``is_primary`` argument is accepted for API symmetry
        with :meth:`MembraneNode.store` but is always overridden
        to ``False`` so a replica cannot accidentally mark
        itself as the canonical owner of a fragment.

        Args:
            fragment: Fragment to store.
            is_primary: Ignored.

        Returns:
            bool: True if the fragment is stored.
        """
        return super().store(fragment, is_primary=False)
