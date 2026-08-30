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
    * **Consistency + HLC (2.0+)**: ``consistency`` selects the write
      level on the cluster boundary; ``hlc`` is the per-fragment
      hybrid logical clock used for conflict resolution.

At 2.0+ the dataclass gained two fields: ``consistency`` (one of
``"strong" | "quorum" | "eventual"``) and ``hlc`` (a 64-bit HLC
integer). :func:`membrane.serialization.to_dict` writes both;
missing keys on the wire trigger :class:`SchemaError` because the
2.0 reader refuses older payloads (no shims).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.hlc import HLC, unpack
from membrane.identity import PayloadIdentity

#: Consistency levels supported on the wire at 2.0+. Producers
#: stamp a default of "strong" at construction time; the
#: :class:`~membrane.server.Server` may reset the level according
#: to the cluster's :attr:`ClusterConfig.default_consistency`
#: at the moment of :func:`op_store`.
_CONSISTENCY_LEVELS: tuple[str, ...] = ("strong", "quorum", "eventual")


@dataclass(frozen=True)
class Fragment:
    """Immutable content-addressed fragment.

    A fragment is the smallest unit of content that Membrane stores,
    transfers, deduplicates, and reconciles. It carries the data's
    identity (:attr:`identity`), a pointer to where the actual bytes
    live (:attr:`payload_ref`), the byte size (:attr:`payload_size`),
    the consistency level (:attr:`consistency`), a hybrid logical
    clock (:attr:`hlc`), and the metadata required for routing and
    lifecycle decisions.

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
        consistency: Write level used by :func:`op_store`:
            ``"strong"`` blocks on quorum ack before responding
            200; ``"quorum"`` blocks on a configurable threshold
            (``floor(replica_count / 2) + 1``); ``"eventual"``
            returns 200 immediately and propagates via the
            asynchronous replication loop. Defaults to ``"strong"``.
        hlc: 64-bit hybrid logical clock integer produced by
            :class:`~membrane.hlc.Clock` at write time. Used for
            conflict resolution during gossip convergence and
            data-migration ordering. Default ``0`` is replaced by
            the writer at every :func:`op_store`.

    Raises:
        ValueError: Raised by :meth:`__post_init__` when any invariant
            (non-negative ``payload_size``/``ttl``, ``reuse_score`` in
            ``[0, 1]``, ``version_id >= 1``, ``consistency`` in
            {strong, quorum, eventual}) is violated.

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
        ...     consistency="strong",
        ...     hlc=0,
        ... )
        >>> frag.identity.payload_hash[:16]
    """

    identity: PayloadIdentity
    payload_ref: str | None
    payload_size: int
    ttl: float
    reuse_score: float
    version_id: int
    consistency: str = "strong"
    hlc: int = 0

    def __post_init__(self) -> None:
        """Validate invariants after construction.

        Runs automatically by the dataclass machinery once ``__init__``
        completes. All checks are O(1) and raise ``ValueError`` with a
        descriptive message on violation rather than relying on
        downstream assertions.

        Raises:
            ValueError: If any of ``payload_size < 0``, ``ttl < 0``,
                ``reuse_score`` outside ``[0, 1]``, ``version_id``
                below 1, or ``consistency`` not in the supported
                levels.
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
        if self.consistency not in _CONSISTENCY_LEVELS:
            raise ValueError(
                f"Fragment consistency must be one of {_CONSISTENCY_LEVELS}, "
                f"got {self.consistency!r}"
            )
        if self.hlc < 0:
            raise ValueError(f"Fragment hlc must be >= 0, got {self.hlc}")

    def hlc_state(self) -> HLC:
        """Decode :attr:`hlc` into its :class:`HLC` components.

        Returns:
            HLC: The decoded (physical_ms, logical) pair.
        """
        return unpack(self.hlc)

    def with_consistency(self, level: str) -> Fragment:
        """Return a copy with ``consistency`` overridden.

        Used by :class:`~membrane.server.Server` to enforce the
        cluster-wide default before :func:`op_store` runs.

        Args:
            level: One of ``"strong"``, ``"quorum"``, ``"eventual"``.

        Returns:
            Fragment: New instance with the updated field.
        """
        if level not in _CONSISTENCY_LEVELS:
            raise ValueError(
                f"Fragment consistency must be one of {_CONSISTENCY_LEVELS}, "
                f"got {level!r}"
            )
        return Fragment(
            identity=self.identity,
            payload_ref=self.payload_ref,
            payload_size=self.payload_size,
            ttl=self.ttl,
            reuse_score=self.reuse_score,
            version_id=self.version_id,
            consistency=level,
            hlc=self.hlc,
        )

    def merge(self, other: Fragment) -> Fragment:
        """Return the fragment with the higher ``hlc``.

        Concurrent writes on the same ``identity`` are resolved
        deterministically by the HLC: the larger clock wins. Ties
        (equal clocks) prefer ``self`` for determinism. ``version_id``
        remains part of the data plane (operational guards like the
        TombstoneTable TTL math) but no longer gates merge
        resolution at 2.0+.

        Args:
            other: The other fragment to merge.

        Returns:
            Fragment: The fragment with the higher ``hlc``.

        Raises:
            ValueError: If ``other.identity != self.identity``.
        """
        if other.identity != self.identity:
            raise ValueError(
                f"cannot merge fragments with different identity: {self.identity} vs {other.identity}"
            )
        if self.hlc >= other.hlc:
            return self
        return Fragment(
            identity=self.identity,
            payload_ref=self.payload_ref or other.payload_ref,
            payload_size=self.payload_size,
            ttl=self.ttl,
            reuse_score=self.reuse_score,
            version_id=self.version_id,
            consistency=self.consistency,
            hlc=other.hlc,
        )


__all__ = ["Fragment"]
