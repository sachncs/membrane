"""ToolTrace: structured tool output as a memory object."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import generate_embedding
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class ToolTrace:
    """A structured output from a tool invocation.

    Attributes:
        tool_name: Name of the invoked tool.
        input_hash: Hash of the tool input/parameters.
        output_hash: Hash of the tool output.
        structured_output: JSON-serializable output string.
        content_hash: Deterministic identity hash.
        semantic_hash: Approximate hash for similarity.
        size_bytes: Storage size.
        reuse_score: Dynamic reuse likelihood.
    """

    tool_name: str
    input_hash: str
    output_hash: str
    structured_output: str
    content_hash: str
    semantic_hash: str
    size_bytes: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize into a Fragment for storage."""
        tokens = tuple(ord(c) for c in self.structured_output)
        embedding = generate_embedding(tokens, 128)
        signature = StructuralSignature(
            model_id="tool",
            layer_range=(0, 0),
            token_span=(0, max(0, len(tokens) - 1)),
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
    def from_fragment(cls, fragment: Fragment) -> "ToolTrace":
        """Reconstruct a ToolTrace from a stored Fragment."""
        return cls(
            tool_name="",
            input_hash="",
            output_hash=fragment.content_hash,
            structured_output="",
            content_hash=fragment.content_hash,
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            reuse_score=fragment.reuse_score,
        )
