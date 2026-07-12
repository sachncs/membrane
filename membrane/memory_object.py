"""MemoryObject: protocol for addressable, comparable, materializable memory.

This module defines :class:`MemoryObject`, the unifying *duck-typed*
protocol for every kind of addressable memory unit in Membrane — be it
a token sequence (:class:`~membrane.prefix.Prefix`), a KV-cache segment
(:class:`~membrane.kv_segment.KVSegment`), an artifact
(:class:`~membrane.artifact.Artifact`), or any future first-class
memory type.

The protocol captures the invariants the rest of the system relies on:

* **Addressable**: Every memory object exposes a ``content_hash`` so it
  can be looked up in the canonical store and across the cluster.
* **Comparable**: Every memory object exposes a ``semantic_hash`` so
  similarity queries can be answered without full content retrieval.
* **Measurable**: ``size_bytes`` and ``token_count`` enable cost and
  capacity accounting.
* **Lifecycle-aware**: ``reuse_score`` biases placement and eviction.
* **Materializable**: Any memory object can be converted into the
  canonical on-disk representation (:class:`~membrane.fragment.Fragment`)
  via :meth:`MemoryObject.materialize`.

Because :class:`MemoryObject` is decorated with
:func:`typing.runtime_checkable`, callers can use ``isinstance(obj,
MemoryObject)`` for cheap structural checks. The protocol is also
reversible: every implementation must provide a
:meth:`MemoryObject.from_fragment` classmethod that round-trips a
:class:`~membrane.fragment.Fragment` back into a typed object.
"""

import logging

logger = logging.getLogger(__name__)


from typing import Protocol, runtime_checkable

from membrane.fragment import Fragment


@runtime_checkable
class MemoryObject(Protocol):
    """Protocol for all first-class memory objects in Membrane.

    Any class that satisfies this structural protocol can be used
    interchangeably by the canonical store, indexes, and routing
    layers. Concrete implementations include
    :class:`~membrane.prefix.Prefix`,
    :class:`~membrane.kv_segment.KVSegment`, and
    :class:`~membrane.artifact.Artifact`.

    Implementations must:
        * Be immutable (so they can safely be shared across threads).
        * Compute ``content_hash`` deterministically from their payload.
        * Provide a :meth:`materialize` method that produces a
          :class:`~membrane.fragment.Fragment` preserving
          ``content_hash`` and the relevant lifecycle fields.
        * Provide a :meth:`from_fragment` classmethod that round-trips
          a stored fragment back into a typed object (with placeholder
          values for fields that fragments cannot represent, such as
          the original token IDs of a prefix).

    Attributes:
        content_hash: Unique hash identifying this object's payload.
            Used as the primary key in the canonical store and in
            cross-node content exchange.
        semantic_hash: Approximate hash of the object's embedding.
            Used by similarity indexes to short-circuit exact-match
            lookups.
        size_bytes: Serialized payload size in bytes.
        token_count: Number of tokens represented by this object. May
            be ``0`` for non-text memory objects (e.g., raw artifacts).
        reuse_score: Producer-supplied reuse likelihood in ``[0, 1]``.
    """

    content_hash: str
    semantic_hash: str
    size_bytes: int
    token_count: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize this memory object into a storable Fragment.

        Returns:
            Fragment: An immutable fragment that captures this
            object's identity, embedding, structural signature, and
            lifecycle metadata. The fragment's ``content_hash`` is
            preserved unchanged.
        """
        ...

    @classmethod
    def from_fragment(cls, fragment: Fragment) -> "MemoryObject":
        """Reconstruct a typed memory object from a stored Fragment.

        Used when the canonical store or a remote peer returns a
        fragment whose original typed object is no longer in memory.
        Fields that fragments cannot represent (e.g., the original
        token sequence of a :class:`~membrane.prefix.Prefix`) are
        recovered as placeholders.

        Args:
            fragment: A fragment previously produced by
                :meth:`materialize` (or with a structurally
                compatible signature).

        Returns:
            MemoryObject: A typed instance of the concrete class,
            with metadata fields preserved and any non-serializable
            fields filled with deterministic placeholders.
        """
        ...
