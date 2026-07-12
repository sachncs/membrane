"""TransferService: inventory exchange, delta sync, and fragment transfer.

This module defines :class:`TransferService`, the in-process
*transfer plane* used by
:class:`~membrane.delta_sync.DeltaSync`,
:class:`~membrane.cluster_replicator.ClusterReplicator`,
:class:`~membrane.origin_node.OriginNode`, and
:class:`~membrane.replica_node.ReplicaNode` to move fragments
between :class:`~membrane.membrane_node.MembraneNode` instances.

The service exposes three primitives:

* :meth:`inventory_digest` — returns a snapshot of the node's
  ``content_hash -> version_id`` mapping.
* :meth:`compare_inventories` — returns the set of hashes that
  are missing or outdated in one digest relative to another.
* :meth:`transfer_fragment` — moves a single fragment from a
  source node to a target node (via their public ``retrieve`` /
  ``store`` methods).

:meth:`sync_nodes` composes the three primitives into a one-shot
*synchronize everything that is newer on source than on target*
operation.

Thread safety:
    The class is stateless; safety is inherited from the
    :class:`MembraneNode` instances passed in.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode


class TransferService:
    """Transfer plane that negotiates and moves fragments between nodes."""

    def __init__(self) -> None:
        """Initialize the transfer service."""
        logger.info("Initialized %s", self.__class__.__name__)
        pass

    def inventory_digest(self, node: MembraneNode) -> dict[str, int]:
        """Build a ``content_hash -> version_id`` digest for ``node``.

        The digest is constructed by iterating over the node's
        public ``fragments`` mapping. It is a snapshot — later
        mutations of the node are not reflected.

        Args:
            node: Node to inventory.

        Returns:
            dict[str, int]: Mapping from content hash to version
            id.
        """
        return {h: frag.version_id for h, frag in node.fragments.items()}

    def compare_inventories(
        self,
        local: dict[str, int],
        remote: dict[str, int],
    ) -> set[str]:
        """Find hashes present in ``remote`` but missing or outdated in ``local``.

        A hash is considered *missing* if it is absent from
        ``local``, and *outdated* if ``local[hash] <
        remote[hash]``. Both contribute to the returned set.

        Args:
            local: Local inventory digest.
            remote: Remote inventory digest.

        Returns:
            set[str]: Set of hashes that should be transferred
            from remote to local.
        """
        missing: set[str] = set()
        for h, remote_version in remote.items():
            local_version = local.get(h)
            if local_version is None or local_version < remote_version:
                missing.add(h)
        return missing

    def transfer_fragment(
        self,
        source: MembraneNode,
        target: MembraneNode,
        content_hash: str,
    ) -> bool:
        """Copy a fragment from ``source`` to ``target``.

        Uses :meth:`MembraneNode.retrieve` on the source and
        :meth:`MembraneNode.store` on the target. The target
        receives the fragment as a non-primary replica.

        Args:
            source: Node holding the fragment.
            target: Node to receive the fragment.
            content_hash: Hash of the fragment to transfer.

        Returns:
            bool: True if the transfer succeeded, False if the
            source did not have the fragment or the target
            rejected the store.
        """
        fragment = source.retrieve(content_hash)
        if fragment is None:
            # Source is missing the fragment (possibly expired or
            # evicted between plan and execution).
            return False
        return target.store(fragment, is_primary=False)

    def sync_nodes(
        self,
        source: MembraneNode,
        target: MembraneNode,
    ) -> list[str]:
        """Synchronize all missing fragments from ``source`` to ``target``.

        Builds both inventory digests, computes the missing set,
        and transfers each missing fragment via
        :meth:`transfer_fragment`. The result list reflects only
        successful transfers; failures are silently skipped.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            list[str]: Content hashes that were successfully
            transferred.
        """
        local = self.inventory_digest(target)
        remote = self.inventory_digest(source)
        missing = self.compare_inventories(local, remote)

        transferred: list[str] = []
        for h in missing:
            if self.transfer_fragment(source, target, h):
                transferred.append(h)
        return transferred
