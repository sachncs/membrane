"""Artifact: retrieved document or embedding as a memory object."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class Artifact:
    """A retrieved document or embedding artifact.

    Attributes:
        source_url: Identifier or URL of the source.
        text_hash: Hash of the raw text content.
        embedding: Dense semantic embedding tuple.
        content_hash: Deterministic identity hash.
        semantic_hash: Approximate hash for similarity.
        size_bytes: Storage size.
        token_count: Token length.
        reuse_score: Dynamic reuse likelihood.
    """

    source_url: str
    text_hash: str
    embedding: tuple[float, ...]
    content_hash: str
    semantic_hash: str
    size_bytes: int
    token_count: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize into a Fragment for storage."""
        signature = StructuralSignature(
            model_id="artifact",
            layer_range=(0, 0),
            token_span=(0, self.token_count - 1),
        )
        return Fragment(
            content_hash=self.content_hash,
            embedding=self.embedding,
            structural_signature=signature,
            size=self.size_bytes,
            ttl=3600.0,
            reuse_score=self.reuse_score,
            version_id=1,
        )

    @classmethod
    def from_fragment(cls, fragment: Fragment) -> "Artifact":
        """Reconstruct an Artifact from a stored Fragment."""
        span = fragment.structural_signature.token_span
        count = span[1] - span[0] + 1
        return cls(
            source_url="",
            text_hash=fragment.content_hash,
            embedding=fragment.embedding,
            content_hash=fragment.content_hash,
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            token_count=count,
            reuse_score=fragment.reuse_score,
        )
