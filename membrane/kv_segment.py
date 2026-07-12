"""KVSegment: per-layer per-head KV cache slice as a memory object.

This module defines :class:`KVSegment`, the canonical addressable unit
for a *physical* KV-cache tensor slice in Membrane. Whereas
:class:`~membrane.prefix.Prefix` represents the logical token sequence
that may be reused, a ``KVSegment`` represents the concrete per-layer
/ per-head KV tensors produced for that prefix by a particular model.

Each segment is independently addressable so that the reconstruction
engine can fetch only the layers it needs (e.g., to fill a single
decoder layer in a prefill-disaggregated serving architecture). The
``content_hash`` is computed over the serialized tensor bytes and
therefore naturally deduplicates identical computations across
requests, models, and nodes.

Lifecycle:
    1. A compute backend (CPU/GPU/Transformers) produces a tensor for
       a specific ``(layer, head, token_span)``.
    2. The backend wraps the tensor in a :class:`KVSegment` with a
       content hash and registers it with the canonical store.
    3. The fragment store and indexes make it discoverable to other
       nodes.
    4. The reconstruction engine retrieves individual segments when
       assembling a full KV cache for a new request.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import generate_embedding
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class KVSegment:
    """A per-layer per-head KV cache slice.

    A segment captures the smallest independently reusable KV-cache
    unit: the cache for a single attention head in a single layer,
    covering a contiguous span of token positions. By decomposing a
    full KV cache into segments, Membrane can:

    * Cache and reuse individual layers across requests with
      partially-overlapping contexts.
    * Transfer only the layers a decoder node needs.
    * Apply eviction at fine granularity.

    Attributes:
        layer: Transformer layer index this segment belongs to. Must
            be non-negative and match the producing model.
        head: Attention head index within the layer. Reserved for
            future per-head sharding; current backends emit segments
            that span all heads for a layer.
        token_span: Inclusive ``(start, end)`` token position range
            covered by this segment within the originating prompt.
        tensor_shape: Original tensor shape as
            ``(heads, seq_len, head_dim)``. Preserved for
            deserialization; not used for content addressing.
        content_hash: Deterministic hash of the serialized tensor
            bytes. Two segments with the same hash are byte-identical
            and interchangeable across nodes.
        semantic_hash: Approximate hash used for similarity lookups
            in the semantic index.
        size_bytes: Serialized payload size in bytes.
        reuse_score: Producer-supplied reuse likelihood in ``[0, 1]``;
            higher values bias placement toward hotter tiers.

    Example:
        >>> seg = KVSegment(
        ...     layer=12,
        ...     head=0,
        ...     token_span=(0, 127),
        ...     tensor_shape=(32, 128, 128),
        ...     content_hash="abc123",
        ...     semantic_hash="abc1",
        ...     size_bytes=32 * 128 * 128 * 2,
        ...     reuse_score=0.9,
        ... )
        >>> frag = seg.materialize()
    """

    layer: int
    head: int
    token_span: tuple[int, int]
    tensor_shape: tuple[int, ...]
    content_hash: str
    semantic_hash: str
    size_bytes: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize this segment into a storable :class:`Fragment`.

        The embedding is generated from the tensor shape so that
        shape-equivalent segments cluster together in the semantic
        index. The structural signature uses ``model_id="kv"`` and a
        single-element ``layer_range`` to mark the fragment as a
        KV-segment rather than a prefix-level fragment.

        Returns:
            Fragment: An immutable fragment carrying the segment's
            ``content_hash``, shape-derived embedding, layer/span
            signature, and lifecycle metadata.
        """
        # The embedding is derived from the shape so that segments
        # with the same shape end up near each other in the
        # semantic index — a useful heuristic for layer-aware
        # locality.
        embedding = generate_embedding(tuple(self.tensor_shape), 128)
        # The synthetic "kv" model_id distinguishes segment-level
        # fragments from prefix-level fragments in the indexes.
        signature = StructuralSignature(
            model_id="kv",
            layer_range=(self.layer, self.layer),
            token_span=self.token_span,
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
    def from_fragment(cls, fragment: Fragment) -> "KVSegment":
        """Reconstruct a :class:`KVSegment` from a stored :class:`Fragment`.

        Used when a fragment is retrieved from the store whose
        origin is a KV segment. The original tensor data is not
        recoverable from the fragment — only its identity, span,
        and metadata.

        Args:
            fragment: A fragment previously produced by
                :meth:`materialize` (or with a structurally
                compatible signature).

        Returns:
            KVSegment: A segment with placeholder ``head`` and
            ``tensor_shape`` and preserved ``content_hash``,
            ``token_span``, ``size_bytes``, and ``reuse_score``.
        """
        layer_range = fragment.structural_signature.layer_range
        return cls(
            layer=layer_range[0],
            # head and tensor_shape cannot be reconstructed from
            # the fragment; use placeholders that keep the segment
            # usable as a structural key.
            head=0,
            token_span=fragment.structural_signature.token_span,
            tensor_shape=(1, 1, 1),
            content_hash=fragment.content_hash,
            # semantic_hash is not preserved on Fragment; reuse the
            # content_hash as a stable surrogate.
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            reuse_score=fragment.reuse_score,
        )
