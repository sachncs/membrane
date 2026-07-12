"""Immutable Fragment data model.

This module defines :class:`Fragment`, the core content-addressed unit of
storage in Membrane. A fragment represents an opaque, immutable chunk of data
(typically a serialized KV-cache tensor slice) together with the metadata
required for routing, placement, deduplication, and lifecycle management.

Fragments follow a value-content addressing scheme: two fragments with the
same ``content_hash`` are considered byte-identical regardless of where or
when they were created. This invariant underpins the canonical store,
cross-node deduplication, and content-based gossip replication used by the
rest of the system.

Design rationale:
    * **Immutability**: Fragments are ``frozen=True`` dataclasses. Once a
      fragment is constructed its identity and metadata cannot drift, which
      is essential for safe sharing across threads, processes, and nodes.
    * **Content addressing**: ``content_hash`` is the primary key in every
      index and store. It is computed by
      :func:`membrane.fragmentation_engine.compute_content_hash`.
    * **Lifecycle metadata**: ``ttl`` and ``reuse_score`` drive eviction and
      promotion decisions made by
      :class:`membrane.promotion_policy.PromotionPolicy`.

Limitations:
    * The dataclass holds a *reference* to the underlying bytes; the actual
      payload is owned by the
      :class:`membrane.fragment_store.FragmentStore` or remote peer.
    * ``reuse_score`` is treated as opaque metadata here. Its semantics are
      defined by the producer (e.g., telemetry or predictor output).
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class Fragment:
    """Immutable content-addressed fragment.

    A fragment is the smallest unit of content that Membrane stores,
    transfers, deduplicates, and reconciles. It carries both the data's
    identity (``content_hash``) and the metadata required for routing and
    lifecycle decisions.

    Instances are hashable and equality-comparable on all fields; this is
    what enables content-based deduplication and safe use as dictionary
    keys or set members throughout the codebase.

    Attributes:
        content_hash: Hex digest uniquely identifying the payload. Two
            fragments with the same hash are considered byte-identical and
            may be deduplicated.
        embedding: Dense vector representation of the fragment content,
            typically produced by :func:`membrane.fragmentation_engine
            .generate_embedding`. Used for semantic indexing.
        structural_signature: Model, layer range, and token span metadata
            describing the fragment's origin and position within a model
            computation graph.
        size: Payload size in bytes. Used for capacity accounting and
            transfer-cost estimation. Must satisfy ``size >= 0``.
        ttl: Time-to-live in seconds before the fragment becomes eligible
            for eviction. ``0`` means "never expires explicitly" but the
            fragment is still subject to LRU/policy-based eviction.
        reuse_score: Producer-supplied reuse likelihood in ``[0, 1]``.
            Higher values indicate the fragment is expected to be reused
            and should therefore be retained and preferentially replicated.
        version_id: Monotonic counter incremented on every content update.
            Together with ``content_hash`` this enables correct cache
            invalidation when underlying data changes. Must be ``>= 1``.

    Raises:
        ValueError: Raised by :meth:`__post_init__` when any invariant
            (non-negative ``size``/``ttl``, ``reuse_score`` in ``[0, 1]``,
            ``version_id >= 1``) is violated.

    Example:
        >>> from membrane.fragment import Fragment
        >>> from membrane.structural_signature import StructuralSignature
        >>> sig = StructuralSignature(
        ...     model_id="llama-3-8b",
        ...     layer_range=(0, 32),
        ...     token_span=(0, 128),
        ...     schema_version=1,
        ... )
        >>> frag = Fragment(
        ...     content_hash="deadbeef" * 8,
        ...     embedding=(0.1, 0.2, 0.3),
        ...     structural_signature=sig,
        ...     size=4096,
        ...     ttl=3600.0,
        ...     reuse_score=0.87,
        ...     version_id=1,
        ... )
        >>> frag.content_hash
    """

    content_hash: str
    embedding: tuple[float, ...]
    structural_signature: StructuralSignature
    size: int
    ttl: float
    reuse_score: float
    version_id: int

    def __post_init__(self) -> None:
        """Validate invariants after construction.

        Runs automatically by the dataclass machinery once ``__init__``
        completes. All checks are O(1) and raise ``ValueError`` with a
        descriptive message on violation rather than relying on
        downstream assertions.

        Raises:
            ValueError: If any of ``size < 0``, ``ttl < 0``,
                ``reuse_score`` outside ``[0, 1]``, or
                ``version_id < 1``.
        """
        if self.size < 0:
            raise ValueError(f"Fragment size must be >= 0, got {self.size}")
        if self.ttl < 0:
            raise ValueError(f"Fragment ttl must be >= 0, got {self.ttl}")
        if not 0.0 <= self.reuse_score <= 1.0:
            raise ValueError(
                f"Fragment reuse_score must be in [0, 1], got {self.reuse_score}"
            )
        if self.version_id < 1:
            raise ValueError(
                f"Fragment version_id must be >= 1, got {self.version_id}"
            )
