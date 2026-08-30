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
    * **Compatibility fingerprint (2.0+)**: The optional
      :attr:`fingerprint_compat` field carries the
      :class:`~membrane.compat.ModelCompatibilityFingerprint`
      digest so operators that swap model archives catch the
      mismatch before bytes are imported into the wrong engine.
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
    clock (:attr:`hlc`), the v2.0+ compatibility fingerprint
    (:attr:`fingerprint_compat`), and the metadata required for
    routing and lifecycle decisions.

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
        fingerprint_compat: 64-character hex digest of the
            :class:`~membrane.compat.ModelCompatibilityFingerprint`
            that produced the bytes. ``""`` when the v2.0+
            compatibility layer is not active (e.g. tests
            synthesising a Fragment by hand). On :func:`op_store`
            the v2 server fills this in from the active compute
            backend's adapter.
    """

    identity: PayloadIdentity
    payload_ref: str | None
    payload_size: int
    ttl: float
    reuse_score: float
    version_id: int
    consistency: str = "strong"
    hlc: int = 0
    fingerprint_compat: str = ""

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
        # ``fingerprint_compat`` is either the empty string
        # (legacy / tests) or a 64-character hex digest. Anything
        # else indicates a wire-side bug.
        if self.fingerprint_compat and (
            len(self.fingerprint_compat) != 64
            or not all(c in "0123456789abcdef" for c in self.fingerprint_compat)
        ):
            raise ValueError(
                "fingerprint_compat must be the empty string or a 64-char hex digest, "
                f"got {self.fingerprint_compat!r}"
            )

    def hlc_state(self) -> HLC:
        """Decode :attr:`hlc` into its :class:`HLC` components.

        Returns:
            HLC: The decoded (physical_ms, logical) pair.
        """
        return unpack(self.hlc)

    def with_consistency(self, level: str) -> "Fragment":
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
            fingerprint_compat=self.fingerprint_compat,
        )

    def with_fingerprint(self, fingerprint: str) -> "Fragment":
        """Return a copy with ``fingerprint_compat`` overridden.

        Args:
            fingerprint: 64-character hex digest or empty string.

        Returns:
            Fragment: New instance with the updated field.
        """
        return Fragment(
            identity=self.identity,
            payload_ref=self.payload_ref,
            payload_size=self.payload_size,
            ttl=self.ttl,
            reuse_score=self.reuse_score,
            version_id=self.version_id,
            consistency=self.consistency,
            hlc=self.hlc,
            fingerprint_compat=fingerprint,
        )

    def merge(self, other: "Fragment") -> "Fragment":
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
            fingerprint_compat=other.fingerprint_compat or self.fingerprint_compat,
        )


__all__ = ["Fragment"]
