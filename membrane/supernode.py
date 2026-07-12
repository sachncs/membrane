"""Supernode: routing table maintainer and directory super-peer.

This module defines :class:`Supernode`, the lightweight directory
service that aggregates fragment-location information for a slice
of the cluster. A deployment typically runs several supernodes,
each covering a subset of the hash ring; together they form a
*hierarchical directory* that the
:class:`~membrane.distributed_directory.DistributedDirectory`
queries in parallel.

The supernode is intentionally minimal:

* It records where fragments live (:meth:`register_fragment`,
  :meth:`unregister_fragment`).
* It answers "who holds this fragment?" queries
  (:meth:`resolve`).
* It maintains a consistent :class:`~membrane.hash_ring.HashRing`
  so it can fall back to placement-based answers when gossip
  state is incomplete (:meth:`resolve_via_ring`).

Thread safety:
    The class is **not thread-safe**. Provide external locking
    when sharing across threads.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.hash_ring import HashRing


class Supernode:
    """Maintains routing tables and resolves fragment locations.

    Acts as a directory super-peer that knows which nodes hold
    which fragments.

    Attributes:
        supernode_id: Stable identifier for this supernode.
        hash_ring: Consistent hash ring used for placement-based
            fallbacks.
        fragment_locations: Mapping from ``content_hash`` to the
            set of node IDs holding a replica.
    """

    def __init__(self, supernode_id: str, hash_ring: HashRing | None = None) -> None:
        """Initialize the supernode.

        Args:
            supernode_id: Unique identifier for this supernode.
            hash_ring: Optional consistent hash ring for node
                placement. A default empty ring is created when
                ``None``.
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

        If the holder set becomes empty after removal, the entry
        is deleted from the directory to prevent stale records
        from accumulating.

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
            set[str]: Defensive copy of the holder set. Empty if
            the supernode has no record of the hash.
        """
        return set(self.fragment_locations.get(content_hash, set()))

    def resolve_via_ring(self, content_hash: str) -> str | None:
        """Return the primary node responsible via the hash ring.

        Useful when the supernode has not yet observed a
        registration for the hash and a placement-based answer is
        needed immediately.

        Args:
            content_hash: Hash to look up.

        Returns:
            str | None: Node identifier from the hash ring, or
            ``None`` if the ring is empty.
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
