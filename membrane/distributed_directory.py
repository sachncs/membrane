"""DistributedDirectory: client-facing fragment location resolution."""

import logging

logger = logging.getLogger(__name__)


from membrane.hash_ring import HashRing
from membrane.supernode import Supernode


class DistributedDirectory:
    """Distributed directory that resolves where fragments live.

    Combines supernode lookups with consistent hashing for O(1) resolution.
    """

    def __init__(self, hash_ring: HashRing | None = None) -> None:
        """Initialize the directory.

        Args:
            hash_ring: Optional hash ring for distributed placement.
        """
        """Initialize the directory.

        Args:
            hash_ring: Optional hash ring for distributed placement.
        """
        self.hash_ring = hash_ring or HashRing()
        self.supernodes: dict[str, Supernode] = {}

    def register_supernode(self, supernode: Supernode) -> None:
        """Register a supernode with this directory.

        Args:
            supernode: Supernode to add.
        """
        self.supernodes[supernode.supernode_id] = supernode

    def locate(self, content_hash: str) -> set[str]:
        """Find all nodes that hold a fragment.

        Args:
            content_hash: Hash to locate.

        Returns:
            Set of node identifiers across all supernodes.
        """
        result: set[str] = set()
        for sn in self.supernodes.values():
            result.update(sn.resolve(content_hash))
        return result

    def locate_nearest(self, content_hash: str, from_node: str) -> str | None:
        """Find the nearest node holding a fragment, falling back to the ring.

        Args:
            content_hash: Hash to locate.
            from_node: Reference node for proximity (not yet used).

        Returns:
            Node identifier, or None if unresolved.
        """
        holders = self.locate(content_hash)
        if holders:
            return min(holders)  # deterministic tie-break
        return self.hash_ring.get_node(content_hash)

    def add_node(self, node_id: str) -> None:
        """Add a node to the underlying hash ring.

        Args:
            node_id: Node identifier.
        """
        self.hash_ring.add_node(node_id)
        for sn in self.supernodes.values():
            sn.add_node(node_id)
