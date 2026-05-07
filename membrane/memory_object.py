"""MemoryObject: protocol for addressable, comparable, materializable memory."""

import logging

logger = logging.getLogger(__name__)


from typing import Protocol, runtime_checkable

from membrane.fragment import Fragment


@runtime_checkable
class MemoryObject(Protocol):
    """Protocol for all first-class memory objects in Membrane.

    A MemoryObject is addressable by content hash, comparable by semantic
    hash, and can be materialized into a canonical Fragment for storage.
    """

    content_hash: str
    semantic_hash: str
    size_bytes: int
    token_count: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize this memory object into a storable Fragment."""
        ...

    @classmethod
    def from_fragment(cls, fragment: Fragment) -> "MemoryObject":
        """Reconstruct a typed memory object from a stored Fragment."""
        ...
