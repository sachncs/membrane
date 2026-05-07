"""Supernode: routing table maintainer and directory super-peer."""

import logging

logger = logging.getLogger(__name__)


from membrane.hash_ring import HashRing


class Supernode:
    """Maintains routing tables and resolves fragment locations.

    Acts as a directory super-peer that knows which nodes hold which fragments.
    """

    def __init__(self, supernode_id: str, hash_ring: HashRing | None = None) -> None:
        """Initialize the supernode.

        Args:
            supernode_id: Unique identifier for this supernode.
            hash_ring: Optional consistent hash ring for node placement.
        """
        """Initialize the supernode.

        Args:
            supernode_id: Unique identifier for this supernode.
            hash_ring: Optional consistent hash ring for node placement.
        """
        self.supernode_id = supernode_id
        self.hash_ring = hash_ring or HashRing()
        self.fragment_locations: dict[str, set[str]] = {}

    def register_fragment(self, content_hash: str, node_id: str) -> None:
        """Record that a fragment is stored on a node.

        Args:
            content_hash: Fragment content hash.
            node_id: Node holding the fragment.
        """
        self.fragment_locations.setdefault(content_hash, set()).add(node_id)

    def unregister_fragment(self, content_hash: str, node_id: str) -> None:
        """Remove a fragment location record.

        Args:
            content_hash: Fragment content hash.
            node_id: Node no longer holding the fragment.
        """
        locs = self.fragment_locations.get(content_hash)
        if locs is not None:
            locs.discard(node_id)
            if not locs:
                del self.fragment_locations[content_hash]

    def resolve(self, content_hash: str) -> set[str]:
        """Return all node IDs that hold the given fragment.

        Args:
            content_hash: Hash to resolve.

        Returns:
            Set of node identifiers. Empty if unknown.
        """
        return set(self.fragment_locations.get(content_hash, set()))

    def resolve_via_ring(self, content_hash: str) -> str | None:
        """Return the primary node responsible for a content hash via the ring.

        Args:
            content_hash: Hash to look up.

        Returns:
            Node identifier from the hash ring, or None if empty.
        """
        return self.hash_ring.get_node(content_hash)

    def add_node(self, node_id: str) -> None:
        """Add a node to the consistent hash ring.

        Args:
            node_id: Node identifier.
        """
        self.hash_ring.add_node(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the consistent hash ring.

        Args:
            node_id: Node identifier.
        """
        self.hash_ring.remove_node(node_id)
