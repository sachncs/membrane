"""DeltaEncoder: compute differences between similar prefixes for incremental sync."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragmentation_engine import compute_content_hash


@dataclass(frozen=True)
class Delta:
    """Delta between two token sequences.

    Attributes:
        base_content_hash: Hash of the base prefix.
        appended_tokens: Tokens added after the base.
        removed_tail_count: Tokens removed from the base tail.
    """

    base_content_hash: str
    appended_tokens: tuple[int, ...]
    removed_tail_count: int


class DeltaEncoder:
    """Computes delta between two similar token sequences."""

    def encode(
        self,
        base_tokens: tuple[int, ...],
        new_tokens: tuple[int, ...],
    ) -> Delta:
        """Compute delta from base to new tokens.

        Args:
            base_tokens: Original token sequence.
            new_tokens: Modified token sequence.

        Returns:
            Delta describing the change.
        """
        base_len = len(base_tokens)
        new_len = len(new_tokens)

        # Find longest common prefix
        common = 0
        for i in range(min(base_len, new_len)):
            if base_tokens[i] != new_tokens[i]:
                break
            common += 1

        removed = base_len - common
        appended = new_tokens[common:]

        return Delta(
            base_content_hash=compute_content_hash(base_tokens),
            appended_tokens=appended,
            removed_tail_count=removed,
        )

    def decode(
        self,
        base_tokens: tuple[int, ...],
        delta: Delta,
    ) -> tuple[int, ...]:
        """Reconstruct new tokens from base + delta.

        Args:
            base_tokens: Original token sequence.
            delta: Delta to apply.

        Returns:
            Reconstructed token tuple.
        """
        kept = base_tokens[: len(base_tokens) - delta.removed_tail_count]
        return kept + delta.appended_tokens
