"""DistributedDirectory: client-facing fragment location resolution.

This module defines :class:`DistributedDirectory`, the lightweight
front-end that application code uses to answer "where does this
fragment live?" queries across a Membrane cluster.

The directory combines two lookup strategies:

* **Supernode lookup** — :class:`~membrane.supernode.Supernode`
  instances each maintain a per-shard view of fragment locations.
  Iterating over every supernode yields the *complete* set of
  holders across the cluster.
* **Hash-ring fallback** — when no supernode reports a holder, the
  directory falls back to the consistent hash ring to determine the
  canonical placement. This keeps writes idempotent even when the
  gossip state has not yet propagated.

Together, the two strategies let clients find a holder without
needing to query every node directly.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.hash_ring import HashRing
from membrane.supernode import Supernode


class DistributedDirectory:
    """Distributed directory that resolves where fragments live.

    Combines supernode lookups with consistent hashing for O(1)
    resolution.

    Attributes:
        hash_ring: HashRing used for placement and as a fallback
            lookup source.
        supernodes: Mapping from ``supernode_id`` to the registered
            :class:`~membrane.supernode.Supernode` instances.
    """

    def __init__(self, hash_ring: HashRing | None = None) -> None:
        """Initialize the directory.

        Args:
            hash_ring: Optional hash ring for distributed
                placement. A default empty ring is created when
                ``None``.
        """
        self.hash_ring = hash_ring or HashRing()
        self.supernodes: dict[str, Supernode] = {}

    def register_supernode(self, supernode: Supernode) -> None:
        """Register a supernode with this directory.

        Args:
            supernode: Supernode to add. Indexed by its
                ``supernode_id`` attribute; re-registering an
                existing id overwrites the previous entry.
        """
        self.supernodes[supernode.supernode_id] = supernode

    def locate(self, content_hash: str) -> set[str]:
        """Find all nodes that hold a fragment.

        Queries every registered supernode and unions the results.
        The returned set may include replicas, primaries, and any
        nodes that have reported holding the fragment via gossip.

        Args:
            content_hash: Hash to locate.

        Returns:
            set[str]: Node identifiers across all supernodes.
            Empty when no supernode has any record of the hash.
        """
        result: set[str] = set()
        for sn in self.supernodes.values():
            result.update(sn.resolve(content_hash))
        return result

    def locate_nearest(self, content_hash: str, from_node: str) -> str | None:
        """Find the nearest node holding a fragment, falling back to the ring.

        The current implementation uses lexical ordering of node
        identifiers as a deterministic tie-breaker. The ``from_node``
        argument is reserved for future proximity-based selection
        (e.g., latency-aware routing) and is not yet used.

        Args:
            content_hash: Hash to locate.
            from_node: Reference node for proximity (not yet used).

        Returns:
            str | None: Node identifier, or ``None`` if both the
            supernodes and the hash ring are empty.
        """
        holders = self.locate(content_hash)
        if holders:
            # Lexical min is a stable, side-effect-free tie-breaker
            # that lets callers reproduce results deterministically.
            return min(holders)
        # Fall back to the ring's canonical owner when no gossip
        # state is available yet (e.g., right after a fresh
        # deployment).
        return self.hash_ring.get_node(content_hash)

    def add_node(self, node_id: str) -> None:
        """Add a node to the underlying hash ring and every supernode.

        Args:
            node_id: Node identifier.
        """
        self.hash_ring.add_node(node_id)
        for sn in self.supernodes.values():
            sn.add_node(node_id)
