"""KVSegment: per-layer per-head KV cache slice as a memory object."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import generate_embedding
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class KVSegment:
    """A per-layer per-head KV cache slice.

    Attributes:
        layer: Layer index.
        head: Attention head index.
        token_span: Inclusive token range (start, end).
        tensor_shape: Shape tuple (heads, seq_len, dim).
        content_hash: Deterministic hash of tensor bytes.
        semantic_hash: Approximate hash for similarity.
        size_bytes: Storage size in bytes.
        reuse_score: Dynamic reuse likelihood.
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
        """Materialize into a Fragment for storage."""
        embedding = generate_embedding(tuple(self.tensor_shape), 128)
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
        """Reconstruct a KVSegment from a stored Fragment."""
        layer_range = fragment.structural_signature.layer_range
        return cls(
            layer=layer_range[0],
            head=0,
            token_span=fragment.structural_signature.token_span,
            tensor_shape=(1, 1, 1),
            content_hash=fragment.content_hash,
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            reuse_score=fragment.reuse_score,
        )
