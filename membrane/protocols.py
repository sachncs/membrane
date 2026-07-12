"""Protocols for core Membrane interfaces.

This module defines the structural :class:`typing.Protocol` interfaces
that the rest of the codebase depends on. By coding against these
protocols rather than concrete classes, individual subsystems can be
swapped (for example, in tests or in alternative deployments) without
rewriting their consumers.

The protocols cover four concerns:

* :class:`IndexProtocol` — the multi-modal fragment index (exact,
  semantic, positional, co-access).
* :class:`RouterProtocol` — selection of a placement node for a
  fragment.
* :class:`DirectoryProtocol` — node membership and fragment
  location tracking.
* :class:`TransportProtocol` — moving fragments between nodes.

All protocols are decorated with
:func:`typing.runtime_checkable` so that ``isinstance`` checks can be
used in tests and debugging paths.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membrane.fragment import Fragment


@runtime_checkable
class IndexProtocol(Protocol):
    """Abstract index supporting exact, semantic, and positional lookup.

    The protocol captures the union of operations provided by the
    four in-memory indexes used in Membrane: exact, semantic,
    positional, and co-access. Concrete implementations include
    :class:`~membrane.exact_index.ExactIndex`,
    :class:`~membrane.semantic_index.SemanticIndex`,
    :class:`~membrane.positional_index.PositionalIndex`, and
    :class:`~membrane.co_access_index.CoAccessIndex`, as well as the
    aggregate :class:`~membrane.index_system.IndexSystem`.

    Implementations are expected to be safe to call concurrently from
    a single writer thread and multiple reader threads; concurrent
    writers are not guaranteed.
    """

    def insert(self, fragment: Fragment, locations: set[str]) -> None:
        """Insert ``fragment`` and record ``locations`` as replicas.

        Args:
            fragment: Fragment to index.
            locations: Set of node IDs that hold a copy of the
                fragment.
        """
        ...

    def remove(self, content_hash: str) -> bool:
        """Remove a fragment from the index.

        Args:
            content_hash: Identifier of the fragment to remove.

        Returns:
            bool: True if the fragment was present and removed,
            False if it was not indexed.
        """
        ...

    def exact_lookup(self, content_hash: str) -> object | None:
        """Look up a fragment by its exact ``content_hash``.

        Args:
            content_hash: Identifier of the fragment.

        Returns:
            object | None: The indexed fragment, or ``None`` if no
            match was found. The concrete return type is determined
            by the implementation.
        """
        ...

    def semantic_lookup(self, embedding: tuple[float, ...], k: int = 3) -> list[Fragment]:
        """Return up to ``k`` fragments nearest to ``embedding``.

        Args:
            embedding: Dense query vector.
            k: Maximum number of results to return. Implementations
                may return fewer if the index is sparse.

        Returns:
            list[Fragment]: Fragments ordered by approximate
            similarity, most similar first.
        """
        ...

    def positional_lookup(self, start: int, end: int) -> list[Fragment]:
        """Return fragments whose token span overlaps ``[start, end]``.

        Args:
            start: Inclusive start position.
            end: Inclusive end position.

        Returns:
            list[Fragment]: Fragments overlapping the requested
            range, in implementation-defined order.
        """
        ...

    def positional_adjacent(self, position: int, max_gap: int = 0) -> list[Fragment]:
        """Return fragments whose token span ends within ``max_gap`` of ``position``.

        Used by the reconstruction engine to find the nearest cached
        fragment when the requested position is not itself cached.

        Args:
            position: Token position of interest.
            max_gap: Maximum acceptable gap (in tokens) between the
                fragment's last position and ``position``.

        Returns:
            list[Fragment]: Adjacent fragments in implementation
            order.
        """
        ...

    def record_co_access(self, a: str, b: str) -> None:
        """Record that two fragments were accessed together.

        Feeds the co-access graph that powers link-prediction
        routing and prefetch hints.

        Args:
            a: First fragment's content hash.
            b: Second fragment's content hash.
        """
        ...

    def co_access_neighbors(self, content_hash: str) -> set[str]:
        """Return the set of fragments frequently accessed with ``content_hash``.

        Args:
            content_hash: Identifier of the fragment to query.

        Returns:
            set[str]: Content hashes of neighboring fragments. Empty
            if the fragment has no recorded co-accesses.
        """
        ...


@runtime_checkable
class RouterProtocol(Protocol):
    """Abstract router that selects a target node for a fragment.

    Implementations encapsulate the placement policy used by the
    system. Concrete examples include
    :class:`~membrane.latency_router.LatencyRouter`,
    :class:`~membrane.economic_router.EconomicRouter`, and the
    :class:`~membrane.joint_optimizer.JointOptimizer`.

    A router is invoked whenever the system needs to decide *where*
    a fragment should live (initial placement, replication, or
    migration).
    """

    def route(
        self,
        fragment: Fragment,
        candidate_node_ids: list[str],
        telemetry_map: dict,
        access_history: list[str],
    ) -> str:
        """Select the best node for ``fragment`` among the candidates.

        Args:
            fragment: The fragment being placed.
            candidate_node_ids: Eligible node IDs. May be a subset
                of the cluster; the router does not need to consider
                nodes outside this list.
            telemetry_map: Per-node telemetry snapshots (load,
                bandwidth, queue depth, etc.). Schema is
                implementation-defined.
            access_history: Recent access history for the calling
                session; may be empty.

        Returns:
            str: The selected node's identifier. The router must
            always return an element of ``candidate_node_ids``; it
            is the caller's responsibility to ensure the candidate
            list is non-empty.
        """
        ...


@runtime_checkable
class DirectoryProtocol(Protocol):
    """Abstract directory for node and fragment location tracking.

    Implementations map fragment ``content_hash`` values to the set
    of node IDs that hold a replica, and node IDs to their current
    membership status. Concrete implementations include
    :class:`~membrane.distributed_directory.DistributedDirectory` and
    :class:`~membrane.global_directory.GlobalDirectory`.
    """

    def register_node(self, node) -> None:
        """Add ``node`` to the directory's membership table.

        Args:
            node: Node-like object exposing at least an identifier
                attribute.
        """
        ...

    def unregister_node(self, node_id: str) -> bool:
        """Remove ``node_id`` from the membership table.

        Args:
            node_id: Identifier of the node to remove.

        Returns:
            bool: True if the node was registered and is now gone,
            False if it was not present.
        """
        ...

    def record_fragment_location(self, content_hash: str, node_id: str) -> None:
        """Record that ``node_id`` holds a replica of ``content_hash``.

        Args:
            content_hash: Identifier of the fragment.
            node_id: Identifier of the node holding a replica.
        """
        ...

    def locate_fragment(self, content_hash: str) -> set[str]:
        """Return the set of node IDs holding a replica.

        Args:
            content_hash: Identifier of the fragment.

        Returns:
            set[str]: Node IDs with a replica. Empty when the
            fragment is unknown to the directory.
        """
        ...


@runtime_checkable
class TransportProtocol(Protocol):
    """Abstract transport for moving fragments between nodes.

    Implementations encapsulate the network-level mechanics of
    moving fragment payloads and reconciling node state. Concrete
    implementations include
    :class:`~membrane.network.peer_client.PeerClient` and
    :class:`~membrane.transfer_service.TransferService`.
    """

    def transfer_fragment(self, source, target, content_hash: str) -> bool:
        """Transfer the fragment with ``content_hash`` from ``source`` to ``target``.

        Args:
            source: Source node (must expose a ``retrieve`` method).
            target: Target node (must expose ``retrieve`` and
                ``store``).
            content_hash: Identifier of the fragment to transfer.

        Returns:
            bool: True if the transfer succeeded, False otherwise
            (e.g., source did not have the fragment or the transfer
            was rejected).
        """
        ...

    def sync_nodes(self, source, target) -> list[str]:
        """Synchronize fragment state from ``source`` to ``target``.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            list[str]: Content hashes that were successfully
            synchronized.
        """
        ...