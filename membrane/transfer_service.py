"""TransferService: inventory exchange, delta sync, and fragment transfer."""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode


class TransferService:
    """Transfer plane that negotiates and moves fragments between nodes."""

    def __init__(self) -> None:
        """Initialize the transfer service."""
        """Initialize the transfer service."""
        logger.info("Initialized %s", self.__class__.__name__)
        pass

    def inventory_digest(self, node: MembraneNode) -> dict[str, int]:
        """Build a content_hash-to-version_id digest for a node.

        Args:
            node: Node to inventory.

        Returns:
            Dictionary mapping content hash to version_id.
        """
        return {h: frag.version_id for h, frag in node.fragments.items()}

    def compare_inventories(
        self,
        local: dict[str, int],
        remote: dict[str, int],
    ) -> set[str]:
        """Find hashes that are present in remote but absent or outdated in local.

        Args:
            local: Local inventory digest.
            remote: Remote inventory digest.

        Returns:
            Set of missing content hashes.
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
        """Copy a fragment from source to target.

        Args:
            source: Node holding the fragment.
            target: Node to receive the fragment.
            content_hash: Hash of the fragment to transfer.

        Returns:
            True if transfer succeeded, else False.
        """
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.store(fragment, is_primary=False)

    def sync_nodes(
        self,
        source: MembraneNode,
        target: MembraneNode,
    ) -> list[str]:
        """Synchronize all missing fragments from source to target.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            List of transferred content hashes.
        """
        local = self.inventory_digest(target)
        remote = self.inventory_digest(source)
        missing = self.compare_inventories(local, remote)

        transferred: list[str] = []
        for h in missing:
            if self.transfer_fragment(source, target, h):
                transferred.append(h)
        return transferred
