"""ShardManager: consistent-hash shard ownership and replica tracking.

Maps content-hash ranges to nodes using a HashRing, tracks primary
owners and replica sets per shard, and supports rebalancing when
the topology changes.

The manager maintains two internal maps:

* ``primary_map`` — ``content_hash -> primary_node_id``. Always
  reflects the current state of the underlying
  :class:`~membrane.hash_ring.HashRing`.
* ``replica_map`` — ``content_hash -> set[replica_node_id]``. The
  primary itself is *not* in this set; replicas are the
  ``replica_count`` distinct nodes that follow the primary on the
  ring.

Both maps are populated lazily by :meth:`assign_shard` and refreshed
in bulk by :meth:`rebalance`. After any topology change
(:meth:`add_node` or :meth:`remove_node`) the manager rebalances
automatically; callers do not need to invoke :meth:`rebalance`
themselves.

Limitations:
    * The maps grow unboundedly with the number of distinct
      ``content_hash`` values seen. For long-running deployments
      consider pruning entries that no longer correspond to active
      fragments.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.hash_ring import HashRing


class ShardManager:
    """Manages shard-to-node assignments via consistent hashing.

    Each content hash is mapped to a primary node by the HashRing.
    Replicas are assigned to the next ``replica_count`` distinct
    nodes in the ring.

    Args:
        hash_ring: HashRing instance for node distribution.
        replica_count: Number of replicas per shard (default 2).
    """

    def __init__(
        self,
        hash_ring: HashRing | None = None,
        replica_count: int = 2,
    ) -> None:
        """Initialize the manager with an optional hash ring.

        Args:
            hash_ring: HashRing to use for node selection. A
                default empty ring is created when ``None``.
            replica_count: Number of replicas per shard (the
                primary itself is excluded from this count).
        """
        self.hash_ring = hash_ring or HashRing()
        self.replica_count = replica_count
        self.primary_map: dict[str, str] = {}
        self.replica_map: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Topology changes
    # ------------------------------------------------------------------

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring and trigger rebalancing.

        Args:
            node_id: Identifier of the new node.
        """
        self.hash_ring.add_node(node_id)
        self.rebalance()

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring and trigger rebalancing.

        Args:
            node_id: Identifier of the node to remove.
        """
        self.hash_ring.remove_node(node_id)
        self.rebalance()

    # ------------------------------------------------------------------
    # Shard resolution
    # ------------------------------------------------------------------

    def assign_shard(self, content_hash: str) -> str:
        """Return the primary node for ``content_hash``, caching the result.

        Args:
            content_hash: Hash to resolve.

        Returns:
            str: Primary node identifier.
        """
        if content_hash not in self.primary_map:
            primary = self.hash_ring.get_node(content_hash)
            self.primary_map[content_hash] = primary
            # Ask the ring for one extra node so the slice beyond
            # the primary can be used as the replica set.
            nodes = self.hash_ring.get_nodes(content_hash, n=self.replica_count + 1)
            # nodes[0] is the primary; the rest are replicas.
            self.replica_map[content_hash] = set(nodes[1:])
        return self.primary_map[content_hash]

    def get_replicas(self, content_hash: str) -> set[str]:
        """Return the set of replica nodes for ``content_hash``.

        Args:
            content_hash: Hash to look up.

        Returns:
            set[str]: Replica node identifiers. Empty when the
            hash has not been assigned.
        """
        # assign_shard populates the replica_map lazily.
        self.assign_shard(content_hash)
        return set(self.replica_map.get(content_hash, set()))

    def get_all_nodes(self, content_hash: str) -> set[str]:
        """Return primary + replicas for ``content_hash``.

        Args:
            content_hash: Hash to look up.

        Returns:
            set[str]: All responsible node identifiers (primary
            and replicas).
        """
        primary = self.assign_shard(content_hash)
        replicas = self.get_replicas(content_hash)
        return {primary} | replicas

    def is_primary(self, content_hash: str, node_id: str) -> bool:
        """Check whether ``node_id`` owns the primary for ``content_hash``.

        Args:
            content_hash: Hash to check.
            node_id: Node to test.

        Returns:
            bool: True if the node is the primary owner.
        """
        return self.assign_shard(content_hash) == node_id

    def is_replica(self, content_hash: str, node_id: str) -> bool:
        """Check whether ``node_id`` holds a replica for ``content_hash``.

        Args:
            content_hash: Hash to check.
            node_id: Node to test.

        Returns:
            bool: True if the node is in the replica set.
        """
        return node_id in self.get_replicas(content_hash)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self) -> list[tuple[str, str, str]]:
        """Recompute all shard assignments after a topology change.

        Iterates over every cached assignment, looks up the new
        primary in the (possibly mutated) hash ring, and updates
        the maps accordingly. Shards whose primary is unchanged
        are skipped.

        Returns:
            list[tuple[str, str, str]]: One tuple per migrated
            shard, ``(content_hash, old_primary, new_primary)``.
        """
        migrations: list[tuple[str, str, str]] = []
        for h, old_primary in list(self.primary_map.items()):
            new_primary = self.hash_ring.get_node(h)
            if new_primary != old_primary:
                migrations.append((h, old_primary, new_primary))
                self.primary_map[h] = new_primary
                nodes = self.hash_ring.get_nodes(h, n=self.replica_count + 1)
                self.replica_map[h] = set(nodes[1:])
        if migrations:
            logger.info("Rebalanced %s shards", len(migrations))
        return migrations

    def shards_for_node(self, node_id: str) -> set[str]:
        """Return all content hashes whose primary is on ``node_id``.

        Args:
            node_id: Node to query.

        Returns:
            set[str]: Content hashes for which ``node_id`` is the
            primary owner.
        """
        return {
            h for h, primary in self.primary_map.items() if primary == node_id
        }

    def replica_shards_for_node(self, node_id: str) -> set[str]:
        """Return all content hashes replicated on ``node_id``.

        Args:
            node_id: Node to query.

        Returns:
            set[str]: Content hashes for which ``node_id`` appears
            in the replica set.
        """
        return {
            h for h, replicas in self.replica_map.items() if node_id in replicas
        }
