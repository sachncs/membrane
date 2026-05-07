"""Protocols for core Membrane interfaces.

Defines typing.Protocol abstractions so that consumers can depend on
interfaces rather than concrete implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membrane.fragment import Fragment


@runtime_checkable
class IndexProtocol(Protocol):
    """Abstract index supporting exact, semantic, and positional lookup."""

    def insert(self, fragment: Fragment, locations: set[str]) -> None: ...
    def remove(self, content_hash: str) -> bool: ...
    def exact_lookup(self, content_hash: str) -> object | None: ...
    def semantic_lookup(self, embedding: tuple[float, ...], k: int = 3) -> list[Fragment]: ...
    def positional_lookup(self, start: int, end: int) -> list[Fragment]: ...
    def positional_adjacent(self, position: int, max_gap: int = 0) -> list[Fragment]: ...
    def record_co_access(self, a: str, b: str) -> None: ...
    def co_access_neighbors(self, content_hash: str) -> set[str]: ...


@runtime_checkable
class RouterProtocol(Protocol):
    """Abstract router that selects a target node for a fragment."""

    def route(
        self,
        fragment: Fragment,
        candidate_node_ids: list[str],
        telemetry_map: dict,
        access_history: list[str],
    ) -> str: ...


@runtime_checkable
class DirectoryProtocol(Protocol):
    """Abstract directory for node and fragment location tracking."""

    def register_node(self, node) -> None: ...
    def unregister_node(self, node_id: str) -> bool: ...
    def record_fragment_location(self, content_hash: str, node_id: str) -> None: ...
    def locate_fragment(self, content_hash: str) -> set[str]: ...


@runtime_checkable
class TransportProtocol(Protocol):
    """Abstract transport for moving fragments between nodes."""

    def transfer_fragment(self, source, target, content_hash: str) -> bool: ...
    def sync_nodes(self, source, target) -> list[str]: ...
