"""Immutable Fragment data model."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class Fragment:
    """Immutable content-addressed fragment.

    Attributes:
        content_hash: Unique hash of fragment content.
        embedding: Dense vector representation.
        structural_signature: Model, layer, and token span info.
        size: Size in bytes (must be >= 0).
        ttl: Time-to-live in seconds (must be >= 0).
        reuse_score: Score in [0, 1] indicating reuse likelihood.
        version_id: Monotonic version counter (must be >= 1).
    """

    content_hash: str
    embedding: tuple[float, ...]
    structural_signature: StructuralSignature
    size: int
    ttl: float
    reuse_score: float
    version_id: int

    def __post_init__(self) -> None:
        """Validate invariants after construction."""
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
