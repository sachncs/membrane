"""ShardManager: consistent-hash shard ownership and replica tracking.

Maps content-hash ranges to nodes using a HashRing, tracks primary owners
and replica sets per shard, and supports rebalancing when the topology
changes.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.hash_ring import HashRing


class ShardManager:
    """Manages shard-to-node assignments via consistent hashing.

    Each content hash is mapped to a primary node by the HashRing.
    Replicas are assigned to the next ``replica_count`` distinct nodes
    in the ring.

    Args:
        hash_ring: HashRing instance for node distribution.
        replica_count: Number of replicas per shard (default 2).
    """

    def __init__(
        self,
        hash_ring: HashRing | None = None,
        replica_count: int = 2,
    ) -> None:
        self.hash_ring = hash_ring or HashRing()
        self.replica_count = replica_count
        self.primary_map: dict[str, str] = {}
        self.replica_map: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Topology changes
    # ------------------------------------------------------------------

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring and trigger rebalancing."""
        self.hash_ring.add_node(node_id)
        self.rebalance()

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring and trigger rebalancing."""
        self.hash_ring.remove_node(node_id)
        self.rebalance()

    # ------------------------------------------------------------------
    # Shard resolution
    # ------------------------------------------------------------------

    def assign_shard(self, content_hash: str) -> str:
        """Return the primary node for a content hash, caching the result.

        Args:
            content_hash: Hash to resolve.

        Returns:
            Primary node identifier.
        """
        if content_hash not in self.primary_map:
            primary = self.hash_ring.get_node(content_hash)
            self.primary_map[content_hash] = primary
            nodes = self.hash_ring.get_nodes(content_hash, n=self.replica_count + 1)
            # nodes[0] is primary; the rest are replicas
            self.replica_map[content_hash] = set(nodes[1:])
        return self.primary_map[content_hash]

    def get_replicas(self, content_hash: str) -> set[str]:
        """Return the set of replica nodes for a content hash.

        Args:
            content_hash: Hash to look up.

        Returns:
            Set of replica node identifiers.
        """
        self.assign_shard(content_hash)
        return set(self.replica_map.get(content_hash, set()))

    def get_all_nodes(self, content_hash: str) -> set[str]:
        """Return primary + replicas for a content hash.

        Args:
            content_hash: Hash to look up.

        Returns:
            Set of all responsible node identifiers.
        """
        primary = self.assign_shard(content_hash)
        replicas = self.get_replicas(content_hash)
        return {primary} | replicas

    def is_primary(self, content_hash: str, node_id: str) -> bool:
        """Check whether *node_id* owns the primary for *content_hash*.

        Args:
            content_hash: Hash to check.
            node_id: Node to test.

        Returns:
            True if the node is the primary owner.
        """
        return self.assign_shard(content_hash) == node_id

    def is_replica(self, content_hash: str, node_id: str) -> bool:
        """Check whether *node_id* holds a replica for *content_hash*.

        Args:
            content_hash: Hash to check.
            node_id: Node to test.

        Returns:
            True if the node holds a replica.
        """
        return node_id in self.get_replicas(content_hash)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self) -> list[tuple[str, str, str]]:
        """Recompute all shard assignments after a topology change.

        Returns:
            List of (content_hash, old_primary, new_primary) for every
            shard whose primary moved.
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
        """Return all content hashes whose primary is on *node_id*.

        Args:
            node_id: Node to query.

        Returns:
            Set of content hashes.
        """
        return {
            h for h, primary in self.primary_map.items() if primary == node_id
        }

    def replica_shards_for_node(self, node_id: str) -> set[str]:
        """Return all content hashes replicated on *node_id*.

        Args:
            node_id: Node to query.

        Returns:
            Set of content hashes.
        """
        return {
            h for h, replicas in self.replica_map.items() if node_id in replicas
        }
