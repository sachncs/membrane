"""Prefix: token sequence memory object.

This module defines :class:`Prefix`, the canonical representation of a
sequence of token IDs in Membrane. A ``Prefix`` is treated as a
first-class memory object: it is content-addressable, semantically
hashed, and can be materialized into a :class:`~membrane.fragment.Fragment`
for storage in the fragment store.

In the broader Membrane design, a prefix represents a *logical* unit of
memory (a sequence of tokens that may be reused across requests). The
companion :class:`~membrane.kv_segment.KVSegment` represents the
*physical* KV-cache tensors produced for that prefix by a particular
model. The two are linked through ``content_hash`` and the model's
structural signature.

Typical lifecycle:
    1. A tokenizer produces an ``(tokens, content_hash)`` pair.
    2. The prefix is registered with the canonical store and indexed
       by the exact and positional indexes.
    3. When a request arrives, the prefix is materialized into a
       :class:`~membrane.fragment.Fragment` whose underlying bytes are
       the actual KV tensors (handled by the
       :class:`~membrane.reconstruction_engine.ReconstructionEngine`).
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import generate_embedding
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class Prefix:
    """A token sequence as a first-class memory object.

    A prefix captures the immutable token sequence that defines a
    reusable context. It is the *logical* identity of a context and is
    independent of any specific model's KV representation.

    The class is frozen and hashable, allowing it to be safely used
    as a dictionary key, in sets, and as the input to content-hash
    computations.

    Attributes:
        tokens: Immutable tuple of integer token IDs that uniquely
            identifies this prefix. Identity is defined by this tuple.
        content_hash: Deterministic cryptographic hash of the token
            sequence. Computed by
            :func:`membrane.fragmentation_engine.compute_content_hash`.
        semantic_hash: Approximate (LSH-style) hash of the embedding
            of the tokens. Used by
            :class:`~membrane.semantic_index.SemanticIndex` for
            similarity lookups without exposing the raw embedding.
        size_bytes: Estimated serialized storage footprint of the
            prefix metadata and (eventually) its KV payload.
        token_count: Cached length of ``tokens``. Redundant with
            ``len(tokens)`` but stored explicitly so that prefix
            metadata can be inspected without materializing the
            token sequence.
        reuse_score: Producer-supplied estimate of how likely this
            prefix is to be reused, in ``[0, 1]``. Higher scores
            bias the canonical store to retain and replicate the
            prefix more aggressively.

    Example:
        >>> from membrane.prefix import Prefix
        >>> from membrane.fragmentation_engine import (
        ...     compute_content_hash,
        ...     generate_embedding,
        ... )
        >>> tokens = (101, 202, 303)
        >>> content_hash = compute_content_hash(b"|".join(str(t).encode() for t in tokens))
        >>> embedding = generate_embedding(tokens, dim=8)
        >>> p = Prefix(
        ...     tokens=tokens,
        ...     content_hash=content_hash,
        ...     semantic_hash=content_hash[:16],
        ...     size_bytes=len(tokens) * 4,
        ...     token_count=len(tokens),
        ...     reuse_score=0.5,
        ... )
        >>> frag = p.materialize()
    """

    tokens: tuple[int, ...]
    content_hash: str
    semantic_hash: str
    size_bytes: int
    token_count: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize this prefix into a storable :class:`Fragment`.

        Produces a fragment whose embedding is derived from the
        token sequence and whose structural signature identifies the
        span as belonging to the synthetic ``"prefix"`` model. The
        returned fragment is the *logical* fragment used by the
        canonical store; the actual KV tensors for the prefix are
        produced separately by the reconstruction engine.

        Returns:
            Fragment: A new immutable fragment representing the
            prefix. ``version_id`` is always ``1`` for fresh
            prefixes; subsequent revisions should produce a new
            fragment with an incremented version.

        Note:
            The fragment's ``content_hash`` is propagated unchanged
            from the prefix so that ``Fragment`` and ``Prefix`` are
            interchangeable as cache keys.
        """
        # Embedding is generated from the token sequence at a fixed
        # dimensionality so that downstream indexes can rely on a
        # stable vector size.
        embedding = generate_embedding(self.tokens, 128)
        # The synthetic "prefix" model id marks the fragment as a
        # logical (not model-specific) unit of memory. Real KV
        # fragments produced by a compute backend carry the actual
        # model id in their StructuralSignature.
        signature = StructuralSignature(
            model_id="prefix",
            layer_range=(0, 0),
            token_span=(0, self.token_count - 1),
        )
        return Fragment(
            content_hash=self.content_hash,
            embedding=embedding,
            structural_signature=signature,
            size=self.size_bytes,
            ttl=3600.0,
            reuse_score=self.reuse_score,
            version_id=1,
        )

    @classmethod
    def from_fragment(cls, fragment: Fragment) -> "Prefix":
        """Reconstruct a :class:`Prefix` from a stored :class:`Fragment`.

        Used when the canonical store (or remote peer) returns a
        fragment whose origin is known to be a prefix but whose
        original ``Prefix`` object is no longer in memory.

        Note:
            The reconstructed ``tokens`` field is a placeholder
            range rather than the original IDs, because fragments do
            not persist token sequences. The reconstructed prefix
            is intended for metadata lookups and routing decisions
            — not for re-tokenizing the original prompt.

        Args:
            fragment: A fragment produced by :meth:`materialize` (or
                with a structurally compatible signature).

        Returns:
            Prefix: A prefix with placeholder tokens but preserved
            ``content_hash``, ``semantic_hash``, ``size_bytes``,
            ``token_count``, and ``reuse_score``.
        """
        span = fragment.structural_signature.token_span
        # The token span is inclusive on both ends; convert to count.
        count = span[1] - span[0] + 1
        return cls(
            # Token sequence is not preserved in fragments; we
            # synthesize a placeholder range so the prefix is still
            # usable as a structural key (count, hashes, etc.).
            tokens=tuple(range(count)),
            content_hash=fragment.content_hash,
            # semantic_hash is not stored on Fragment; reuse the
            # content_hash prefix as a stable surrogate.
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            token_count=count,
            reuse_score=fragment.reuse_score,
        )


# Re-exported for convenience: callers can import Delta alongside Prefix
# when implementing prefix deltas / chain updates. The noqa is scoped
# narrowly to F401 (imported but unused) because Delta is intentionally
# re-exported; the E402 (import not at top of file) is intentional so
# the comment block above can describe the re-export.
from membrane.delta_encoder import Delta  # noqa: F401, E402
