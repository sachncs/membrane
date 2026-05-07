"""Prefix: token sequence memory object."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import generate_embedding
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class Prefix:
    """A token sequence as a first-class memory object.

    Attributes:
        tokens: Immutable tuple of token IDs.
        content_hash: Deterministic hash of the token sequence.
        semantic_hash: Approximate hash of the embedding.
        size_bytes: Estimated storage size.
        token_count: Number of tokens.
        reuse_score: Dynamic reuse likelihood in [0, 1].
    """

    tokens: tuple[int, ...]
    content_hash: str
    semantic_hash: str
    size_bytes: int
    token_count: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize into a Fragment for storage."""
        embedding = generate_embedding(self.tokens, 128)
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
        """Reconstruct a Prefix from a stored Fragment."""
        span = fragment.structural_signature.token_span
        count = span[1] - span[0] + 1
        return cls(
            tokens=tuple(range(count)),
            content_hash=fragment.content_hash,
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            token_count=count,
            reuse_score=fragment.reuse_score,
        )


# Use Delta from delta_encoder for prefix deltas as well.
from membrane.delta_encoder import Delta  # noqa: F401
