"""Immutable Fragment data model.

This module defines :class:`Fragment`, the core content-addressed unit
of storage in Membrane. A fragment represents an immutable chunk of
data (typically a serialized KV-cache tensor slice) together with the
metadata required for routing, placement, deduplication, and lifecycle
management.

Fragments follow a value-content addressing scheme: two fragments with
the same :class:`~membrane.identity.PayloadIdentity` are considered
byte-identical regardless of where or when they were created. This
invariant underpins the canonical store, cross-node deduplication, and
content-based gossip replication used by the rest of the system.

Design rationale:
    * **Identity, not hash**: The full :class:`PayloadIdentity` is the
      primary key in every index and store. ``identity.payload_hash``
      covers *what* the bytes are; the remaining nine fields cover
      *where* in the computation the bytes belong and on which
      model/tokenizer/revision/dtype the bytes can be consumed.
    * **Payload reference, not bytes**: The fragment dataclass holds a
      ``payload_ref`` pointer (the blob key in the canonical
      :class:`~membrane.persistence.ContentStore`) and a
      ``payload_size``. The actual bytes never cross the Fragment
      boundary.
    * **Immutability**: Fragments are ``frozen=True`` dataclasses.
      Once constructed, identity, payload reference, and metadata
      cannot drift, which is essential for safe sharing across
      threads, processes, and nodes.
    * **Lifecycle metadata**: ``ttl`` and ``reuse_score`` drive
      eviction and promotion decisions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.identity import PayloadIdentity


@dataclass(frozen=True)
class Fragment:
    """Immutable content-addressed fragment.

    A fragment is the smallest unit of content that Membrane stores,
    transfers, deduplicates, and reconciles. It carries the data's
    identity (:attr:`identity`), a pointer to where the actual bytes
    live (:attr:`payload_ref`), the byte size (:attr:`payload_size`),
    and the metadata required for routing and lifecycle decisions.

    Instances are hashable and equality-comparable on all fields; this
    is what enables content-based deduplication and safe use as
    dictionary keys or set members throughout the codebase.

    Attributes:
        identity: The :class:`PayloadIdentity` ten-field fingerprint.
            This is the composite key in every index, store, and on
            the wire.
        payload_ref: Blob key under which the canonical frame lives
            in the configured :class:`~membrane.persistence.ContentStore`.
            ``None`` denotes a metadata-only fragment with no payload
            (used for tombstone markers and short-lived operational
            records).
        payload_size: Real byte size of the canonical frame. Must
            satisfy ``payload_size >= 0``. Drives capacity accounting,
            transfer-cost estimation, and replica bandwidth budgets.
        ttl: Time-to-live in seconds before the fragment becomes
            eligible for eviction. ``0`` means "never expires
            explicitly" but the fragment is still subject to
            LRU/policy-based eviction.
        reuse_score: Producer-supplied reuse likelihood in ``[0, 1]``.
            Higher values indicate the fragment is expected to be
            reused and should therefore be retained and preferentially
            replicated.
        version_id: Monotonic counter incremented on every content
            update. Together with :attr:`identity` this enables
            correct cache invalidation when underlying data changes.
            Must be ``>= 1``.

    Raises:
        ValueError: Raised by :meth:`__post_init__` when any invariant
            (non-negative ``payload_size``/``ttl``, ``reuse_score`` in
            ``[0, 1]``, ``version_id >= 1``) is violated.

    Example:
        >>> from membrane.fragment import Fragment
        >>> from membrane.identity import PayloadIdentity
        >>> ident = PayloadIdentity(
        ...     payload_hash="a" * 64,
        ...     model_id="llama-3-8b",
        ...     model_revision="main",
        ...     tokenizer_name="llama-3-8b",
        ...     tokenizer_revision="main",
        ...     layer_range=(0, 32),
        ...     head_range=(-1, -1),
        ...     token_span=(0, 128),
        ...     dtype="float16",
        ...     shape=(1, 32, 32, 128, 64),
        ... )
        >>> frag = Fragment(
        ...     identity=ident,
        ...     payload_ref="a" * 64,
        ...     payload_size=8 * 1024 * 1024,
        ...     ttl=3600.0,
        ...     reuse_score=0.87,
        ...     version_id=1,
        ... )
        >>> frag.identity.payload_hash[:16]
    """

    identity: PayloadIdentity
    payload_ref: str | None
    payload_size: int
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
            ValueError: If any of ``payload_size < 0``, ``ttl < 0``,
                ``reuse_score`` outside ``[0, 1]``, or
                ``version_id < 1``.
        """
        if self.payload_size < 0:
            raise ValueError(f"Fragment payload_size must be >= 0, got {self.payload_size}")
        if self.ttl < 0:
            raise ValueError(f"Fragment ttl must be >= 0, got {self.ttl}")
        if not 0.0 <= self.reuse_score <= 1.0:
            raise ValueError(
                f"Fragment reuse_score must be in [0, 1], got {self.reuse_score}"
            )
        if self.version_id < 1:
            raise ValueError(f"Fragment version_id must be >= 1, got {self.version_id}")

    def merge(self, other: Fragment) -> Fragment:
        """Return the fragment with the higher ``version_id``.

        Used during cluster convergence to resolve concurrent writes
        under the AP merge policy: the larger version wins. Ties
        (equal ``version_id``) prefer ``self`` for determinism.

        Args:
            other: The other fragment to merge with.

        Returns:
            Fragment: The fragment with the higher ``version_id``.

        Raises:
            ValueError: If ``other.identity != self.identity``.
        """
        if other.identity != self.identity:
            raise ValueError(
                f"cannot merge fragments with different identity: {self.identity} vs {other.identity}"
            )
        return self if self.version_id >= other.version_id else other


__all__ = ["Fragment"]
