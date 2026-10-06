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

import logging
from typing import TYPE_CHECKING, Protocol

from membrane.fragment import Fragment
from membrane.node import Node
from membrane.replication import payload_for, replicate_fragment

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
        """``content_hash -> version_id`` for every fragment held.

        Returns:
            dict[str, int]: ``content_hash -> version_id`` for every fragment
            held.
        """
        ...

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Fetch a fragment by hash, or ``None`` if absent.

        Args:
            content_hash: Content hash of the fragment.

        Returns:
            Fragment | None: A fragment by hash, or ``None`` if absent.
        """
        ...

    def payload(self, fragment: Fragment) -> bytes | None:
        """Return the KV bytes behind ``fragment``.

        Args:
            fragment: The fragment.

        Returns:
            bytes | None: The bytes, or ``None`` when metadata-only or absent.
        """
        ...

    def store(self, fragment: Fragment, *, is_primary: bool = False, payload: bytes | None = None) -> bool:
        """Persist ``fragment`` and its bytes; returns True on success.

        Args:
            fragment: The fragment.
            is_primary: Whether this node owns the fragment's primary copy.
            payload: The fragment's KV bytes, written to the content store
                first; required when it has a ``payload_ref`` the store lacks.

        Returns:
            bool: True when stored.
        """
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
        """Inventory digest fetched over the wire, or ``None`` on failure.

        Returns:
            dict[str, int] | None: Inventory digest fetched over the wire, or
            ``None`` on failure.
        """
        ...

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Fetch a fragment over the wire, or ``None`` on failure.

        Args:
            content_hash: Content hash of the fragment.

        Returns:
            Fragment | None: A fragment over the wire, or ``None`` on failure.
        """
        ...

    def payload(self, fragment: Fragment) -> bytes | None:
        """Return the KV bytes behind ``fragment``.

        Args:
            fragment: The fragment.

        Returns:
            bytes | None: The bytes, or ``None`` when metadata-only or absent.
        """
        ...

    def push(self, fragment: Fragment, payload: bytes | None) -> bool:
        """Send a fragment and its bytes to the remote peer.

        Args:
            fragment: The fragment.
            payload: Its KV bytes, when it has a ``payload_ref``.

        Returns:
            bool: True when the peer acknowledged.
        """
        ...


class NodeEndpoint:
    """Adapter that promotes a :class:`Node` to a :class:`LocalEndpoint`."""

    __slots__ = ("node",)

    def __init__(self, node: Node) -> None:
        """Wrap a local :class:`~membrane.node.Node`.

        Args:
            node: Local node.
        """
        self.node = node

    def inventory(self) -> dict[str, int]:
        """Return ``content_hash -> version_id`` for the node's fragments.

        Returns:
            dict[str, int]: ``content_hash -> version_id`` for the node's
            fragments.
        """
        return {h: frag.version_id for h, frag in self.node.fragment_snapshot().items()}

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Return a fragment from the node.

        Args:
            content_hash: Content hash of the fragment.

        Returns:
            Fragment | None: A fragment from the node.
        """
        return self.node.retrieve(content_hash)

    def payload(self, fragment: Fragment) -> bytes | None:
        """Return the fragment's bytes from the node's content store.

        Args:
            fragment: The fragment.

        Returns:
            bytes | None: The bytes, or ``None`` when metadata-only or absent.
        """
        return payload_for(fragment, self.node.content_store)

    def store(self, fragment: Fragment, *, is_primary: bool = False, payload: bytes | None = None) -> bool:
        """Store a fragment, and its bytes, on the node.

        Args:
            fragment: The fragment.
            is_primary: Whether this node owns the fragment's primary copy.
            payload: The fragment's KV bytes.

        Returns:
            bool: True when stored. In-process transfers keep working when
            the source never held bytes (library use without a content
            store); :meth:`Node.store` logs the missing payload. The
            network path refuses such replicas (``POST /replicate``
            returns 422).
        """
        ref = fragment.payload_ref
        if ref is not None and payload is not None and not self.node.content_store.has(ref):
            self.node.content_store.put(ref, payload)
        return self.node.store(fragment, is_primary=is_primary)


class ClusterPeerEndpoint:
    """Adapter that promotes a peer node-id + cluster to a :class:`RemoteEndpoint`."""

    __slots__ = ("cluster", "node_id")

    def __init__(self, node_id: str, cluster: Cluster) -> None:
        """Address peer ``node_id`` through ``cluster``.

        Args:
            node_id: Node identifier.
            cluster: Cluster whose membership resolves the peer.
        """
        self.node_id = node_id
        self.cluster = cluster

    def client_for(self) -> Peer | None:
        """Return the HTTP client for the peer, if it is a member.

        Returns:
            Peer | None: The HTTP client for the peer, if it is a member.
        """
        return self.cluster.membership.get_client(self.node_id)

    def inventory(self) -> dict[str, int] | None:
        """Return the peer's inventory digest.

        Returns:
            dict[str, int] | None: The peer's inventory digest.
        """
        client = self.client_for()
        if client is None:
            return None
        resp = client.get_inventory()
        if not resp:
            return None
        return resp.get("digest", {})

    def retrieve(self, content_hash: str) -> Fragment | None:
        """Return a fragment from the peer.

        Args:
            content_hash: Content hash of the fragment.

        Returns:
            Fragment | None: A fragment from the peer.
        """
        client = self.client_for()
        if client is None:
            return None
        return client.retrieve_fragment(content_hash)

    def payload(self, fragment: Fragment) -> bytes | None:
        """Download the fragment's bytes from the peer (digest-verified).

        Args:
            fragment: The fragment.

        Returns:
            bytes | None: The bytes, or ``None`` when metadata-only or
            unavailable.
        """
        client = self.client_for()
        if client is None or fragment.payload_ref is None:
            return None
        return client.get_blob(fragment.payload_ref)

    def push(self, fragment: Fragment, payload: bytes | None) -> bool:
        """Replicate ``fragment`` and its bytes to the peer.

        Args:
            fragment: The fragment.
            payload: Its KV bytes, when it has a ``payload_ref``.

        Returns:
            bool: True when the peer acknowledged.
        """
        client = self.client_for()
        if client is None:
            return False
        return replicate_fragment(client, fragment, payload)


def resolve_endpoint(node_or_id: Node | str, cluster: Cluster | None) -> LocalEndpoint | RemoteEndpoint:
    """Promote a ``Node`` or remote node-id to the matching endpoint.

    Args:
        node_or_id: A local :class:`~membrane.node.Node`, or the id of a
            peer.
        cluster: Cluster used to resolve peer ids; required for remote ids.

    Returns:
        LocalEndpoint | RemoteEndpoint: The endpoint.
    """
    if isinstance(node_or_id, Node):
        return NodeEndpoint(node_or_id)
    if cluster is None:
        msg = f"remote endpoint {node_or_id!r} requested but no cluster is configured"
        raise ValueError(msg)
    return ClusterPeerEndpoint(node_or_id, cluster)


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

    def resolve_endpoint(self, node_or_id: Node | str) -> LocalEndpoint | RemoteEndpoint:
        """Resolve a node or node id to a local or remote endpoint.

        Args:
            node_or_id: A local :class:`~membrane.node.Node`, or the id of a
                peer.

        Returns:
            LocalEndpoint | RemoteEndpoint: The endpoint.
        """
        return resolve_endpoint(node_or_id, self.cluster_manager)

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
        """Move a fragment between two local endpoints.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Content hash of the fragment.

        Returns:
            bool: True when the fragment was transferred.
        """
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.store(fragment, is_primary=False, payload=source.payload(fragment))

    def transfer_remote_source(
        self,
        source: RemoteEndpoint,
        target: RemoteEndpoint,
        content_hash: str,
    ) -> bool:
        """Fetch a fragment from a remote source and push via the remote target.

        Remote-to-remote transfers chain through the source
        peer's HTTP API: the source serves the fragment over
        GET /retrieve; the target pulls it via POST /replicate.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Content hash of the fragment.

        Returns:
            bool: True when the fragment was transferred.
        """
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.push(fragment, source.payload(fragment))

    def transfer_remote_target(
        self,
        source: LocalEndpoint,
        target: RemoteEndpoint,
        content_hash: str,
    ) -> bool:
        """Read from a local source and push via the remote target's /replicate.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Content hash of the fragment.

        Returns:
            bool: True when the fragment was transferred.
        """
        fragment = source.retrieve(content_hash)
        if fragment is None:
            return False
        return target.push(fragment, source.payload(fragment))

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
        endpoint = self.resolve_endpoint(node)
        result = endpoint.inventory()
        if isinstance(result, dict):
            return result
        return None

    def compare_inventories(
        self,
        local: dict[str, int],
        remote: dict[str, int],
    ) -> set[str]:
        """Find hashes present in ``remote`` but missing or outdated in ``local``.

        Args:
            local: ``content_hash -> version_id`` of the receiving side.
            remote: ``content_hash -> version_id`` of the sending side.

        Returns:
            set[str]: Content hashes ``local`` should fetch.
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
            src_endpoint = self.resolve_endpoint(source)
            tgt_endpoint = self.resolve_endpoint(target)
        except ValueError:
            return False

        if isinstance(src_endpoint, NodeEndpoint) and isinstance(tgt_endpoint, NodeEndpoint):
            return self.transfer_local_endpoint(src_endpoint, tgt_endpoint, content_hash)
        if isinstance(src_endpoint, ClusterPeerEndpoint):
            if isinstance(tgt_endpoint, NodeEndpoint):
                # Pull from a peer into a local node.
                fragment = src_endpoint.retrieve(content_hash)
                if fragment is None:
                    return False
                return tgt_endpoint.store(fragment, payload=src_endpoint.payload(fragment))
            if isinstance(tgt_endpoint, ClusterPeerEndpoint):
                return self.transfer_remote_source(src_endpoint, tgt_endpoint, content_hash)
            return False
        # isinstance(tgt_endpoint, ClusterPeerEndpoint)  (mypy narrowing)
        if not isinstance(src_endpoint, NodeEndpoint) or not isinstance(tgt_endpoint, ClusterPeerEndpoint):
            return False
        return self.transfer_remote_target(
            src_endpoint,
            tgt_endpoint,
            content_hash,
        )

    def sync_nodes(
        self,
        source: Node | str,
        target: Node | str,
    ) -> list[str]:
        """Synchronize all missing fragments from ``source`` to ``target``.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.

        Returns:
            list[str]: Content hashes transferred.
        """
        try:
            src_endpoint = self.resolve_endpoint(source)
            tgt_endpoint = self.resolve_endpoint(target)
        except ValueError:
            return []

        if isinstance(src_endpoint, NodeEndpoint) and isinstance(tgt_endpoint, NodeEndpoint):
            return self.sync_local(src_endpoint.node, tgt_endpoint.node)

        src_digest = src_endpoint.inventory()
        tgt_digest = tgt_endpoint.inventory()
        if not isinstance(src_digest, dict) or not isinstance(tgt_digest, dict):
            return []

        missing = self.compare_inventories(tgt_digest, src_digest)
        transferred: list[str] = []
        for h in missing:
            if isinstance(src_endpoint, NodeEndpoint) and isinstance(tgt_endpoint, NodeEndpoint):
                ok: bool = self.transfer_local_endpoint(src_endpoint, tgt_endpoint, h)
            elif isinstance(src_endpoint, ClusterPeerEndpoint):
                ok = self.transfer_remote_source(
                    src_endpoint,
                    tgt_endpoint,  # type: ignore[arg-type]
                    h,
                )
            elif isinstance(tgt_endpoint, ClusterPeerEndpoint):
                ok = self.transfer_remote_target(
                    src_endpoint,  # type: ignore[arg-type]
                    tgt_endpoint,
                    h,
                )
            else:
                ok = False
            if ok:
                transferred.append(h)
        return transferred

    def sync_local(self, source: Node, target: Node) -> list[str]:
        """Synchronize all missing fragments between two local nodes.

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.

        Returns:
            list[str]: Content hashes transferred.
        """
        local = self.inventory_digest(target) or {}
        remote = self.inventory_digest(source) or {}
        missing = self.compare_inventories(local, remote)
        transferred: list[str] = []
        for h in missing:
            if self.transfer_local(source, target, h):
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

        Args:
            source: Source node or remote node id.
            target: Target node or remote node id.
            content_hash: Content hash of the fragment.

        Returns:
            bool: True when the fragment was transferred.
        """
        return self.transfer_local_endpoint(NodeEndpoint(source), NodeEndpoint(target), content_hash)

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

        Returns:
            bool: True when the fragment was fetched and stored.
        """
        if self.cluster_manager is None:
            return False
        src_endpoint = ClusterPeerEndpoint(source_id, self.cluster_manager)
        try:
            tgt_endpoint = self.resolve_endpoint(target)
        except ValueError:
            return False
        if isinstance(tgt_endpoint, NodeEndpoint):
            fragment = src_endpoint.retrieve(content_hash)
            if fragment is None:
                return False
            return tgt_endpoint.store(fragment, is_primary=False, payload=src_endpoint.payload(fragment))
        if not isinstance(tgt_endpoint, ClusterPeerEndpoint):
            return False
        # Remote target: chain peer-to-peer.
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

        Returns:
            bool: True when the fragment was transferred.
        """
        if self.cluster_manager is None:
            return False
        return self.transfer_remote_target(
            NodeEndpoint(source),
            ClusterPeerEndpoint(target_id, self.cluster_manager),
            content_hash,
        )


__all__ = ["LocalEndpoint", "RemoteEndpoint", "TransferService"]
