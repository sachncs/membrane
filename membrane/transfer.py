"""TransferService: fragment movement between local and remote nodes.

Defines :class:`TransferService` plus two small polymorphism
seams (:class:`LocalEndpoint` and :class:`RemoteEndpoint`)
that let the dispatch between local-to-local, remote-source,
and remote-target transfers happen through behavior rather
than ``isinstance(node, Node)`` branches.

* :class:`LocalEndpoint` — wraps a :class:`~membrane.node.Node`
  so any local node can be addressed as an endpoint.
* :class:`RemoteEndpoint` — wraps a peer node-id and resolves
  to an HTTP client through the cluster's membership table.

:class:`TransferService.transfer_fragment` now selects an
endpoint pair, then dispatches to one of three concrete
operations:

* ``transfer_local_endpoint(source, target, hash)`` — both
  endpoints are local; bytes go Node → Node via
  ``Node.retrieve`` / ``Node.store``.
* ``transfer_remote_source(source, target, hash)`` — source is
  remote; fetch via ``Peer.retrieve_fragment`` and store on
  the local target (or replicate peer-to-peer when both ends
  are remote).
* ``transfer_remote_target(source, target, hash)`` — target is
  remote; read from the local source and push via
  ``Peer.request_replicate``.

The four primitives on :class:`TransferService` continue to
take ``Node | str`` for backward compatibility with callers
that already pass a node or node-id directly; the dispatch
selects the appropriate :class:`LocalEndpoint` /
:Class:`RemoteEndpoint` and then invokes the polymorphic
operation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from membrane.fragment import Fragment
from membrane.node import Node

if TYPE_CHECKING:
    from membrane.network.cluster import Cluster
    from membrane.network.peer import Peer


logger = logging.getLogger(__name__)


class LocalEndpoint(Protocol):
    """Polymorphic local endpoint exposing the operations transfer needs."""

    @property
    def node(self) -> Node:
        """The underlying :class:`Node` instance."""
        ...

    def inventory(self) -> dict[str, int]:
        """``content_hash -> version_id`` for every fragment held."""
        ...

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Fetch a fragment by hash, or ``None`` if absent."""
        ...

    def store(self, fragment: Fragment, *, is_primary: bool = False) -> bool:
        """Persist ``fragment``; returns True on success."""
        ...


class RemoteEndpoint(Protocol):
    """Polymorphic remote endpoint exposing the operations transfer needs.

    Concrete implementations satisfy this Protocol by holding a
    peer node-id and using the cluster to resolve it to the
    HTTP client.
    """

    @property
    def node_id(self) -> str:
        """Stable identifier of the remote peer."""
        ...

    def inventory(self) -> dict[str, int] | None:
        """Inventory digest fetched over the wire, or ``None`` on failure."""
        ...

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Fetch a fragment over the wire, or ``None`` on failure."""
        ...

    def push(self, fragment: Fragment) -> bool:
        """Send a fragment to the remote peer (e.g., via its /replicate endpoint)."""
        ...


class _LocalEndpoint:
    """Adapter that promotes a :class:`Node` to a :class:`LocalEndpoint`."""

    __slots__ = ("node",)

    def __init__(self, node: Node) -> None:
        self.node = node

    def inventory(self) -> dict[str, int]:
        return {h: frag.version_id for h, frag in self.node.fragments.items()}

    def retrieve(self, content_hash: str) -> Fragment | None:
        return self.node.retrieve(content_hash)

    def store(self, fragment: Fragment, *, is_primary: bool = False) -> bool:
        return self.node.store(fragment, is_primary=is_primary)


class _RemoteEndpoint:
    """Adapter that promotes a peer node-id + cluster to a :class:`RemoteEndpoint`."""

    __slots__ = ("node_id", "_cluster")

    def __init__(self, node_id: str, cluster: Cluster) -> None:
        self.node_id = node_id
        self._cluster = cluster

    def _client(self) -> Peer | None:
        return self._cluster.get_peer_client(self.node_id)

    def inventory(self) -> dict[str, int] | None:
        client = self._client()
        if client is None:
            return None
        resp = client.get_inventory()
        if not resp:
            return None
        return resp.get("digest", {})

    def retrieve(self, content_hash: str) -> Fragment | None:
        client = self._client()
        if client is None:
            return None
        return client.retrieve_fragment(content_hash)

    def push(self, fragment: Fragment) -> bool:
        client = self._client()
        if client is None:
            return False
        return client.request_replicate(fragment)


def _resolve(node_or_id: Node | str, cluster: Cluster | None) -> LocalEndpoint | RemoteEndpoint:
    """Promote a ``Node`` or remote node-id to the matching endpoint."""
    if isinstance(node_or_id, Node):
        return _LocalEndpoint(node_or_id)
    if cluster is None:
        msg = f"remote endpoint {node_or_id!r} requested but no cluster is configured"
        raise ValueError(msg)
    return _RemoteEndpoint(node_or_id, cluster)


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
                used to resolve peer clients by node id. When
                ``None``, only local-to-local transfers are supported.
            local_node: The local :class:`~membrane.node.Node`
                instance; used as the default source for outgoing
                remote transfers and as the destination for incoming
                remote transfers.
        """
        self.cluster_manager = cluster_manager
        self.local_node = local_node

    def _resolve_endpoint(self, node_or_id: Node | str) -> LocalEndpoint | RemoteEndpoint:
        return _resolve(node_or_id, self.cluster_manager)

    # ------------------------------------------------------------------
    # Dispatch table — three concrete transfer operations, each
    # implemented as a method on this class. The ``transfer_fragment``
    # entry point selects the right one based on the endpoint kinds
    # rather than chasing isinstance branches.
    # ------------------------------------------------------------------

    def transfer_local_endpoint(
        self,
        source: LocalEndpoint,
        target: LocalEndpoint,
        content_hash: str,
    ) -> bool:
        """Move a fragment between two local endpoints."""
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.store(fragment, is_primary=False)

    def transfer_remote_source(
        self,
        source: RemoteEndpoint,
        target: LocalEndpoint | RemoteEndpoint,
        content_hash: str,
    ) -> bool:
        """Fetch a fragment from a remote source and store on the target."""
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        if isinstance(target, _LocalEndpoint):
            return target.store(fragment, is_primary=False)
        # Remote-to-remote: chain via the source peer's HTTP
        # API by asking the source's /replicate endpoint.
        return target.push(fragment)

    def transfer_remote_target(
        self,
        source: LocalEndpoint,
        target: RemoteEndpoint,
        content_hash: str,
    ) -> bool:
        """Read from a local source and push via the remote target's /replicate."""
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.push(fragment)

    # ------------------------------------------------------------------
    # Public API (Node-or-id convenience wrappers)
    # ------------------------------------------------------------------

    def inventory_digest(self, node: Node | str) -> dict[str, int] | None:
        """Build (or fetch) a ``content_hash -> version_id`` digest.

        Args:
            node: Local node or remote node id.

        Returns:
            dict[str, int] | None: Mapping from content hash to
            version id, or ``None`` when the inventory cannot be
            obtained for a remote node.
        """
        endpoint = self._resolve_endpoint(node)
        result = endpoint.inventory()
        if isinstance(result, dict):
            return result
        return None

    def compare_inventories(
        self,
        local: dict[str, int],
        remote: dict[str, int],
    ) -> set[str]:
        """Find hashes present in ``remote`` but missing or outdated in ``local``."""
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
        string node id (remote). The dispatch selects one of the
        three endpoint-to-endpoint transfer operations based on
        the kind of endpoint produced for source and target.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Hash of the fragment to transfer.

        Returns:
            bool: True on success, False on any failure (missing
            peer client, missing fragment, refused replication).
        """
        try:
            src_endpoint = self._resolve_endpoint(source)
            tgt_endpoint = self._resolve_endpoint(target)
        except ValueError:
            return False

        src_local = isinstance(src_endpoint, _LocalEndpoint)
        tgt_local = isinstance(tgt_endpoint, _LocalEndpoint)
        if src_local and tgt_local:
            return self.transfer_local_endpoint(src_endpoint, tgt_endpoint, content_hash)
        if not src_local:
            return self.transfer_remote_source(
                src_endpoint, tgt_endpoint, content_hash  # type: ignore[arg-type]
            )
        return self.transfer_remote_target(
            src_endpoint, tgt_endpoint, content_hash  # type: ignore[arg-type]
        )

    def sync_nodes(
        self,
        source: Node | str,
        target: Node | str,
    ) -> list[str]:
        """Synchronize all missing fragments from ``source`` to ``target``."""
        try:
            src_endpoint = self._resolve_endpoint(source)
            tgt_endpoint = self._resolve_endpoint(target)
        except ValueError:
            return []

        if isinstance(src_endpoint, _LocalEndpoint) and isinstance(tgt_endpoint, _LocalEndpoint):
            return self.sync_local(src_endpoint.node, tgt_endpoint.node)

        src_digest = src_endpoint.inventory()
        tgt_digest = tgt_endpoint.inventory()
        if not isinstance(src_digest, dict) or not isinstance(tgt_digest, dict):
            return []

        missing = self.compare_inventories(tgt_digest, src_digest)
        transferred: list[str] = []
        src_local = isinstance(src_endpoint, _LocalEndpoint)
        tgt_local = isinstance(tgt_endpoint, _LocalEndpoint)
        for h in missing:
            ok = False
            if src_local and tgt_local:
                ok = self.transfer_local_endpoint(src_endpoint, tgt_endpoint, h)
            elif not src_local:
                ok = self.transfer_remote_source(src_endpoint, tgt_endpoint, h)
            elif not tgt_local:
                ok = self.transfer_remote_target(src_endpoint, tgt_endpoint, h)
            if ok:
                transferred.append(h)
        return transferred

    def sync_local(self, source: Node, target: Node) -> list[str]:
        """Synchronize all missing fragments between two local nodes."""
        local = self.inventory_digest(target) or {}
        remote = self.inventory_digest(source) or {}
        missing = self.compare_inventories(local, remote)
        transferred: list[str] = []
        for h in missing:
            if self.transfer_local(_LocalEndpoint(source), _LocalEndpoint(target), h):
                transferred.append(h)
        return transferred

    def transfer_local(
        self,
        source: Node,
        target: Node,
        content_hash: str,
    ) -> bool:
        """Copy a fragment between two local nodes.

        Convenience wrapper over
        :meth:`transfer_local_endpoint` for callers that pass
        :class:`Node` instances directly.
        """
        return self.transfer_local_endpoint(_LocalEndpoint(source), _LocalEndpoint(target), content_hash)

    def pull_from_remote(
        self,
        source_id: str,
        target: Node | str,
        content_hash: str,
    ) -> bool:
        """Fetch ``content_hash`` from the remote ``source_id`` and store it.

        Args:
            source_id: Remote peer node id (source).
            target: Local :class:`~membrane.node.Node` or remote
                node id (target). When remote, the fragment is
                replicated peer-to-peer via the source peer's
                ``/replicate`` endpoint.
            content_hash: Hash to transfer.
        """
        if self.cluster_manager is None:
            return False
        try:
            src_endpoint = _RemoteEndpoint(source_id, self.cluster_manager)
            tgt_endpoint = self._resolve_endpoint(target)
        except ValueError:
            return False
        return self.transfer_remote_source(src_endpoint, tgt_endpoint, content_hash)

    def push_to_remote(
        self,
        source: Node,
        target_id: str,
        content_hash: str,
    ) -> bool:
        """Read from the local source and push via the remote target peer.

        Args:
            source: Local source node.
            target_id: Remote peer node id (target).
            content_hash: Hash to transfer.
        """
        if self.cluster_manager is None:
            return False
        return self.transfer_remote_target(
            _LocalEndpoint(source),
            _RemoteEndpoint(target_id, self.cluster_manager),
            content_hash,
        )


__all__ = ["TransferService", "LocalEndpoint", "RemoteEndpoint"]
