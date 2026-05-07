"""RemoteTransferService: network-aware fragment transfer.

Delegates local copies to the base ``TransferService`` and remote copies
via ``PeerClient``.
"""

import logging
from typing import TYPE_CHECKING

from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService

if TYPE_CHECKING:
    from membrane.network.cluster_manager import ClusterManager

logger = logging.getLogger(__name__)


class RemoteTransferService(TransferService):
    """Transfer service that handles both local and remote nodes.

    Args:
        cluster_manager: ClusterManager for resolving peer clients.
        local_node: The local MembraneNode instance.
    """

    def __init__(
        self,
        cluster_manager: "ClusterManager | None",
        local_node: MembraneNode,
    ) -> None:
        super().__init__()
        self.cluster_manager = cluster_manager
        self.local_node = local_node

    def transfer_fragment(
        self,
        source: MembraneNode | str,
        target: MembraneNode | str,
        content_hash: str,
    ) -> bool:
        """Copy a fragment from source to target.

        Accepts either a ``MembraneNode`` instance (local) or a ``str``
        node_id (remote).
        """
        # Both local
        if isinstance(source, MembraneNode) and isinstance(target, MembraneNode):
            return super().transfer_fragment(source, target, content_hash)

        if self.cluster_manager is None:
            return False

        # Remote source
        if isinstance(source, str):
            client = self.cluster_manager.get_peer_client(source)
            if client is None:
                logger.warning("No client for source node %s", source)
                return False
            frag = client.retrieve_fragment(content_hash)
            if frag is None:
                return False
            if isinstance(target, MembraneNode):
                return target.store(frag, is_primary=False)
            else:
                t_client = self.cluster_manager.get_peer_client(target)
                if t_client is None:
                    return False
                return t_client.request_replicate(frag)

        # Remote target
        if isinstance(target, str):
            frag = source.retrieve(content_hash)
            if frag is None:
                return False
            client = self.cluster_manager.get_peer_client(target)
            if client is None:
                logger.warning("No client for target node %s", target)
                return False
            return client.request_replicate(frag)

        return False

    def sync_nodes(
        self,
        source: MembraneNode | str,
        target: MembraneNode | str,
    ) -> list[str]:
        """Synchronize all missing fragments from source to target."""
        # Both local
        if isinstance(source, MembraneNode) and isinstance(target, MembraneNode):
            return super().sync_nodes(source, target)

        # Need to get inventories remotely
        source_digest = self._inventory_digest(source)
        target_digest = self._inventory_digest(target)
        if source_digest is None or target_digest is None:
            return []

        missing = self.compare_inventories(target_digest, source_digest)
        transferred: list[str] = []
        for h in missing:
            if self.transfer_fragment(source, target, h):
                transferred.append(h)
        return transferred

    def _inventory_digest(self, node: MembraneNode | str) -> dict[str, int] | None:
        if isinstance(node, MembraneNode):
            return super().inventory_digest(node)
        if self.cluster_manager is None:
            return None
        client = self.cluster_manager.get_peer_client(node)
        if client is None:
            return None
        resp = client.get_inventory()
        if resp:
            return resp.get("digest", {})
        return None
