"""HashRing: consistent hashing for memory ID distribution."""

import bisect
import hashlib
import logging

logger = logging.getLogger(__name__)


class EmptyRingError(ValueError):
    """Raised when an operation is attempted on an empty hash ring."""


class HashRing:
    """Consistent hash ring mapping content hashes to responsible nodes.

    Uses MD5 of node IDs placed at multiple virtual points on a ring.
    Lookups use binary search for O(log n) performance.
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        """Initialize an empty hash ring.

        Args:
            virtual_nodes: Number of virtual points per physical node.
        """
        self.virtual_nodes = virtual_nodes
        self.ring: dict[str, str] = {}
        self.sorted_keys: list[str] = []
        self.node_ids: set[str] = set()

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring.

        Args:
            node_id: Unique node identifier.
        """
        if node_id in self.node_ids:
            return
        self.node_ids.add(node_id)
        for i in range(self.virtual_nodes):
            key = self.hash_value(f"{node_id}:{i}")
            self.ring[key] = node_id
        self.sorted_keys = sorted(self.ring.keys())
        logger.debug("Added node %s to ring", node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring.

        Args:
            node_id: Node identifier to remove.
        """
        if node_id not in self.node_ids:
            return
        self.node_ids.discard(node_id)
        for i in range(self.virtual_nodes):
            key = self.hash_value(f"{node_id}:{i}")
            self.ring.pop(key, None)
        self.sorted_keys = sorted(self.ring.keys())
        logger.debug("Removed node %s from ring", node_id)

    def require_non_empty(self) -> None:
        """Raise EmptyRingError if the ring has no nodes."""
        if not self.sorted_keys:
            raise EmptyRingError(
                "HashRing is empty: add at least one node before lookup"
            )

    def get_node(self, content_hash: str) -> str:
        """Return the node responsible for a content hash.

        Uses binary search for O(log n) lookup.

        Args:
            content_hash: Hash to look up.

        Returns:
            Node identifier.

        Raises:
            EmptyRingError: If the ring has no nodes.
        """
        self.require_non_empty()
        hash_key = self.hash_value(content_hash)
        idx = bisect.bisect_left(self.sorted_keys, hash_key)
        if idx < len(self.sorted_keys):
            return self.ring[self.sorted_keys[idx]]
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]

    def get_nodes(self, content_hash: str, n: int = 3) -> list[str]:
        """Return the top-N nodes responsible for a content hash.

        Args:
            content_hash: Hash to look up.
            n: Number of distinct nodes to return.

        Returns:
            List of unique node identifiers.

        Raises:
            EmptyRingError: If the ring has no nodes.
        """
        self.require_non_empty()
        hash_key = self.hash_value(content_hash)
        nodes: list[str] = []
        seen: set[str] = set()
        start_index = bisect.bisect_left(self.sorted_keys, hash_key)
        for key in self.sorted_keys[start_index:] + self.sorted_keys[:start_index]:
            node = self.ring[key]
            if node not in seen:
                seen.add(node)
                nodes.append(node)
                if len(nodes) >= n:
                    break
        return nodes

    @staticmethod
    def hash_value(value: str) -> str:
        """Compute an MD5 hex digest for ring placement.

        Args:
            value: String to hash.

        Returns:
            Hexadecimal digest string.
        """
        return hashlib.md5(value.encode("utf-8")).hexdigest()
