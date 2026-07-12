"""HashRing: consistent hashing for memory ID distribution.

This module implements a minimal *consistent hash ring* used by
:class:`~membrane.shard_manager.ShardManager` to map
``content_hash`` values to responsible nodes.

Each physical node is replicated across ``virtual_nodes`` points
on the ring (default 150) to improve the uniformity of the
distribution. Adding or removing a node only redistributes the
fraction of keys that fell on the affected virtual nodes —
``O(K / N)`` rather than ``O(K)`` — which is the defining property
of consistent hashing.

Lookups use :func:`bisect.bisect_left` for O(log n) placement of
the query hash on the sorted ring.

Algorithm references:
    * Karger et al., "Consistent Hashing and Random Trees:
      Distributed Caching Protocols for Relieving Hot Spots on the
      World Wide Web", 1997.

Limitations:
    * The hash function is MD5 over the UTF-8 encoding of the
      input. MD5 is not collision-resistant but it is fast and
      provides sufficient distribution uniformity for placement.
    * The ring does not balance load — only ownership. A node may
      hold more virtual points than another if their IDs happen
      to hash non-uniformly. Tune ``virtual_nodes`` if you observe
      significant imbalance.
"""

import bisect
import hashlib
import logging

logger = logging.getLogger(__name__)


class EmptyRingError(ValueError):
    """Raised when an operation is attempted on an empty hash ring.

    Inherits from :class:`ValueError` for backward compatibility
    with code that catches ``ValueError`` on lookup failures.
    """


class HashRing:
    """Consistent hash ring mapping content hashes to responsible nodes.

    Uses MD5 of node IDs placed at multiple virtual points on a
    ring. Lookups use binary search for O(log n) performance.

    Attributes:
        virtual_nodes: Number of virtual points placed per
            physical node. Higher values give better uniformity
            at the cost of more bookkeeping.
        ring: Mapping from ring position (hex digest) to the
            owning node ID.
        sorted_keys: Sorted list of ring positions for binary
            search. Kept in sync with ``ring`` by
            :meth:`add_node` and :meth:`remove_node`.
        node_ids: Set of physical node IDs currently in the ring.
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        """Initialize an empty hash ring.

        Args:
            virtual_nodes: Number of virtual points per physical
                node. Must be positive.
        """
        self.virtual_nodes = virtual_nodes
        self.ring: dict[str, str] = {}
        self.sorted_keys: list[str] = []
        self.node_ids: set[str] = set()

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring.

        Idempotent: re-adding an existing node is a no-op.

        Args:
            node_id: Unique node identifier.
        """
        if node_id in self.node_ids:
            return
        self.node_ids.add(node_id)
        for i in range(self.virtual_nodes):
            key = self.hash_value(f"{node_id}:{i}")
            self.ring[key] = node_id
        # Re-sort once after all virtual points are inserted to
        # avoid O(virtual_nodes * log n) per insertion.
        self.sorted_keys = sorted(self.ring.keys())
        logger.debug("Added node %s to ring", node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring.

        No-op if the node is not in the ring.

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
        """Raise :class:`EmptyRingError` if the ring has no nodes.

        Centralizes the empty-ring check so individual lookup
        methods stay readable.
        """
        if not self.sorted_keys:
            raise EmptyRingError(
                "HashRing is empty: add at least one node before lookup"
            )

    def get_node(self, content_hash: str) -> str:
        """Return the node responsible for ``content_hash``.

        Uses binary search for O(log n) lookup.

        Args:
            content_hash: Hash to look up.

        Returns:
            str: Node identifier of the owner. When the lookup
            position falls past the last ring entry, the result
            wraps to the smallest entry (modular ring).

        Raises:
            EmptyRingError: If the ring has no nodes.
        """
        self.require_non_empty()
        hash_key = self.hash_value(content_hash)
        idx = bisect.bisect_left(self.sorted_keys, hash_key)
        if idx < len(self.sorted_keys):
            return self.ring[self.sorted_keys[idx]]
        # Wrap around to the first node — the ring is circular.
        return self.ring[self.sorted_keys[0]]

    def get_nodes(self, content_hash: str, n: int = 3) -> list[str]:
        """Return the top-N distinct nodes responsible for ``content_hash``.

        Walks the ring clockwise from the lookup position,
        collecting nodes until either ``n`` distinct entries are
        seen or the ring is exhausted.

        Args:
            content_hash: Hash to look up.
            n: Number of distinct nodes to return. Values ``<= 0``
                return an empty list.

        Returns:
            list[str]: Unique node identifiers in ring order,
            starting from the primary owner.

        Raises:
            EmptyRingError: If the ring has no nodes.
        """
        self.require_non_empty()
        if n <= 0:
            return []
        hash_key = self.hash_value(content_hash)
        nodes: list[str] = []
        seen: set[str] = set()
        start_index = bisect.bisect_left(self.sorted_keys, hash_key)
        # Concatenate the post-lookup slice with the pre-lookup
        # slice so the walk wraps around the ring boundary.
        ordered = self.sorted_keys[start_index:] + self.sorted_keys[:start_index]
        for key in ordered:
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
            str: Hexadecimal digest string.
        """
        return hashlib.md5(value.encode("utf-8")).hexdigest()
