"""RemoteTransferService: network-aware fragment transfer.

This module defines :class:`RemoteTransferService`, a subclass of
:class:`~membrane.transfer_service.TransferService` that can move
fragments between a mix of local
:class:`~membrane.membrane_node.MembraneNode` instances and
remote peers (identified by their ``node_id`` string).

Routing rules:

* ``source`` and ``target`` both local → delegate to the base
  :class:`TransferService`.
* ``source`` is a remote id → fetch the fragment via the peer's
  :class:`~membrane.network.peer_client.PeerClient` and store
  it on the local target.
* ``target`` is a remote id → read the fragment locally and push
  it via the peer's ``request_replicate`` verb.

The class is the bridge that lets the rest of the system use a
single transfer API regardless of whether the underlying peer is
in the same process or across the network.
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
        cluster_manager: :class:`ClusterManager` for resolving
            peer clients.
        local_node: The local :class:`MembraneNode` instance.
    """

    def __init__(
        self,
        cluster_manager: "ClusterManager | None",
        local_node: MembraneNode,
    ) -> None:
        """Initialize the service.

        Args:
            cluster_manager: Cluster manager used to resolve
                peer clients by node id. ``None`` disables
                remote transfers.
            local_node: Local node instance; used as the
                default source for outgoing remote transfers.
        """
        super().__init__()
        self.cluster_manager = cluster_manager
        self.local_node = local_node

    def transfer_fragment(
        self,
        source: MembraneNode | str,
        target: MembraneNode | str,
        content_hash: str,
    ) -> bool:
        """Copy a fragment from ``source`` to ``target``.

        Accepts either a :class:`MembraneNode` (local) or a
        string node id (remote). The dispatch logic depends on
        which combination of source/target is local.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Fragment to transfer.

        Returns:
            bool: True if the transfer succeeded, False on any
            failure (missing peer client, missing fragment,
            refused replication).
        """
        # Fast path: both local.
        if isinstance(source, MembraneNode) and isinstance(target, MembraneNode):
            return super().transfer_fragment(source, target, content_hash)

        if self.cluster_manager is None:
            return False

        # Remote source: pull the fragment to the local target.
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
            # Remote-to-remote: replicate from source to target
            # via the source's HTTP API.
            t_client = self.cluster_manager.get_peer_client(target)
            if t_client is None:
                return False
            return t_client.request_replicate(frag)

        # Remote target: push the local fragment over HTTP.
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
        """Synchronize all missing fragments from ``source`` to ``target``.

        For local-to-local pairs this delegates to the base
        class. For pairs that involve remote nodes, it fetches
        both inventories over HTTP, computes the missing set,
        and transfers each missing fragment via
        :meth:`transfer_fragment`.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.

        Returns:
            list[str]: Successfully transferred hashes.
        """
        # Local-to-local fast path.
        if isinstance(source, MembraneNode) and isinstance(target, MembraneNode):
            return super().sync_nodes(source, target)

        # Remote paths: fetch both inventories via HTTP.
        source_digest = self.inventory_digest(source)
        target_digest = self.inventory_digest(target)
        if source_digest is None or target_digest is None:
            return []

        missing = self.compare_inventories(target_digest, source_digest)
        transferred: list[str] = []
        for h in missing:
            if self.transfer_fragment(source, target, h):
                transferred.append(h)
        return transferred

    def inventory_digest(  # type: ignore[override]
        self, node: MembraneNode | str
    ) -> dict[str, int] | None:
        """Fetch a node's inventory digest.

        Args:
            node: Local :class:`MembraneNode` or remote node id.

        Returns:
            dict[str, int] | None: ``content_hash -> version_id``
            for the node, or ``None`` when the inventory cannot
            be obtained (missing client, network failure).
        """
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
