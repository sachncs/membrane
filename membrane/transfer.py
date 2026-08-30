"""TransferService: fragment movement between local and remote nodes.

This module defines :class:`TransferService`, the single transfer-plane
class used by :class:`~membrane.delta_sync.DeltaSync`,
:class:`~membrane.replicator.Replicator`,
:class:`~membrane.origin.Origin`, and
:class:`~membrane.replica.Replica` to move fragments between
:class:`~membrane.node.Node` instances.

The service exposes four primitives:

* :meth:`inventory_digest` — returns a snapshot of the node's
  ``content_hash -> version_id`` mapping, or fetches the same from a
  remote peer.
* :meth:`compare_inventories` — returns the set of hashes that are
  missing or outdated in one digest relative to another.
* :meth:`transfer_fragment` — moves a single fragment from a source
  node to a target node. Either the source or the target (or both)
  may be a remote peer identified by node id; the cluster
  subsystem is used to look up the peer's HTTP client.
* :meth:`sync_nodes` — composes the three primitives into a one-shot
  *synchronize everything that is newer on source than on target*
  operation.

Routing rules:
    * ``source`` and ``target`` both local (Node instances) — delegate
      directly to the local store.
    * ``source`` is a remote id — fetch the fragment via the peer's
      HTTP client and store it on the local target.
    * ``target`` is a remote id — read the fragment locally and push
      it via the peer's ``request_replicate`` verb.

Thread safety:
    The class itself is stateless beyond the optional
    ``cluster_manager`` / ``local_node`` references; safety is
    inherited from the underlying :class:`~membrane.node.Node`
    instances.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from membrane.node import Node

if TYPE_CHECKING:
    from membrane.network.cluster import Cluster


logger = logging.getLogger(__name__)


class TransferService:
    """Transfer plane that negotiates and moves fragments between nodes."""

    def __init__(
        self,
        cluster_manager: Cluster | None = None,
        local_node: Node | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            cluster_manager: :class:`~membrane.network.cluster.Cluster`
                used to resolve peer clients by node id. When ``None``,
                only local-to-local transfers are supported.
            local_node: The local :class:`~membrane.node.Node`
                instance; used as the default source for outgoing
                remote transfers and as the destination for incoming
                remote transfers.
        """
        self.cluster_manager = cluster_manager
        self.local_node = local_node

    # ------------------------------------------------------------------
    # Local primitives
    # ------------------------------------------------------------------

    def inventory_digest(self, node: Node | str) -> dict[str, int] | None:
        """Build (or fetch) a ``content_hash -> version_id`` digest.

        Local :class:`~membrane.node.Node` instances return a fresh
        snapshot; remote node ids return the digest the peer reports
        over HTTP, or ``None`` if no client is configured or the peer
        is unreachable.

        Args:
            node: Local node or remote node id.

        Returns:
            dict[str, int] | None: Mapping from content hash to
            version id, or ``None`` when the inventory cannot be
            obtained for a remote node.
        """
        if isinstance(node, Node):
            return {h: frag.version_id for h, frag in node.fragments.items()}
        if self.cluster_manager is None:
            return None
        client = self.cluster_manager.get_peer_client(node)
        if client is None:
            return None
        resp = client.get_inventory()
        if resp:
            return resp.get("digest", {})
        return None

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
            set[str]: Hashes that should be transferred from
            remote to local.
        """
        missing: set[str] = set()
        for h, remote_version in remote.items():
            local_version = local.get(h)
            if local_version is None or local_version < remote_version:
                missing.add(h)
        return missing

    def transfer_fragment(
        self,
        source: Node | str,
        target: Node | str,
        content_hash: str,
    ) -> bool:
        """Copy a fragment from ``source`` to ``target``.

        Accepts either a :class:`~membrane.node.Node` (local) or a
        string node id (remote). The dispatch depends on which
        combination of source/target is local.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Hash of the fragment to transfer.

        Returns:
            bool: True on success, False on any failure (missing peer
            client, missing fragment, refused replication).
        """
        if isinstance(source, Node) and isinstance(target, Node):
            return self.transfer_local(source, target, content_hash)
        if self.cluster_manager is None:
            return False
        if isinstance(source, str):
            return self.pull_from_remote(source, target, content_hash)
        if isinstance(target, str):
            return self.push_to_remote(source, target, content_hash)
        return False

    def sync_nodes(
        self,
        source: Node | str,
        target: Node | str,
    ) -> list[str]:
        """Synchronize all missing fragments from ``source`` to ``target``.

        Local-to-local pairs delegate to the base implementation.
        Pairs that involve remote nodes fetch both inventories over
        HTTP, compute the missing set, and transfer each missing
        fragment via :meth:`transfer_fragment`.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.

        Returns:
            list[str]: Successfully transferred hashes.
        """
        if isinstance(source, Node) and isinstance(target, Node):
            return self.sync_local(source, target)

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

    # ------------------------------------------------------------------
    # Internal dispatch helpers
    # ------------------------------------------------------------------

    def transfer_local(
        self,
        source: Node,
        target: Node,
        content_hash: str,
    ) -> bool:
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.store(fragment, is_primary=False)

    def sync_local(self, source: Node, target: Node) -> list[str]:
        local = self.inventory_digest(target) or {}
        remote = self.inventory_digest(source) or {}
        missing = self.compare_inventories(local, remote)
        transferred: list[str] = []
        for h in missing:
            if self.transfer_local(source, target, h):
                transferred.append(h)
        return transferred

    def pull_from_remote(
        self,
        source_id: str,
        target: Node | str,
        content_hash: str,
    ) -> bool:
        cluster = self.cluster_manager
        if cluster is None:
            return False
        client = cluster.get_peer_client(source_id)
        if client is None:
            logger.warning("No client for source node %s", source_id)
            return False
        frag = client.retrieve_fragment(content_hash)
        if frag is None:
            return False
        if isinstance(target, Node):
            return target.store(frag, is_primary=False)
        # Remote-to-remote: replicate from source to target via the
        # source's HTTP API.
        t_client = cluster.get_peer_client(target)
        if t_client is None:
            return False
        return t_client.request_replicate(frag)

    def push_to_remote(
        self,
        source: Node,
        target_id: str,
        content_hash: str,
    ) -> bool:
        cluster = self.cluster_manager
        if cluster is None:
            return False
        frag = source.retrieve(content_hash)
        if frag is None:
            return False
        client = cluster.get_peer_client(target_id)
        if client is None:
            logger.warning("No client for target node %s", target_id)
            return False
        return client.request_replicate(frag)


__all__ = ["TransferService"]
