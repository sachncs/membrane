"""DeltaEncoder: compute differences between similar prefixes for incremental sync.

This module implements a minimal *prefix delta* codec: given two
token sequences ``base`` and ``new`` that share a common prefix,
:meth:`DeltaEncoder.encode` produces a :class:`Delta` describing
only the appended tokens and the number of tokens removed from
the tail of ``base``. The matching :meth:`decode` function
reconstructs ``new`` from ``base`` plus the delta.

The codec is intentionally simple — it does not attempt to handle
non-prefix insertions or internal edits. For richer edit
operations, layer a Myers diff or use
:class:`~membrane.prefix_version_chain.PrefixVersionChain` to
track arbitrary lineage.

Use cases:
    * Compact representation of conversation turns that extend a
      prior prefix.
    * Bandwidth-efficient incremental sync between nodes that
      share most of a context.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragmentation_engine import compute_content_hash


@dataclass(frozen=True)
class Delta:
    """Delta between two token sequences.

    Attributes:
        base_content_hash: Content hash of the base prefix, used
            to identify the ancestor when reconstructing.
        appended_tokens: Tokens that follow the shared prefix in
            the new sequence.
        removed_tail_count: Number of tokens dropped from the
            tail of the base sequence before the appended tokens
            take over.
    """

    base_content_hash: str
    appended_tokens: tuple[int, ...]
    removed_tail_count: int


class DeltaEncoder:
    """Computes the prefix delta between two similar token sequences."""

    def encode(
        self,
        base_tokens: tuple[int, ...],
        new_tokens: tuple[int, ...],
    ) -> Delta:
        """Compute the delta from ``base_tokens`` to ``new_tokens``.

        The encoder walks both sequences from the start, recording
        the length of the longest common prefix, and then takes
        everything after that point in ``new_tokens`` as the
        appended tail.

        Args:
            base_tokens: Original token sequence.
            new_tokens: Modified token sequence.

        Returns:
            Delta: Description of the change. ``removed_tail_count``
            equals ``len(base_tokens) - common_prefix_length``,
            regardless of whether ``new_tokens`` is shorter or
            longer than ``base_tokens``.
        """
        base_len = len(base_tokens)
        new_len = len(new_tokens)

        # Find longest common prefix.
        common = 0
        for i in range(min(base_len, new_len)):
            if base_tokens[i] != new_tokens[i]:
                break
            common += 1

        # Tokens removed from the base's tail (may exceed the
        # number of appended tokens, in which case decode will
        # produce a sequence shorter than the base).
        removed = base_len - common
        # Tokens appended after the shared prefix.
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
        """Reconstruct ``new_tokens`` from ``base_tokens`` + ``delta``.

        The decoder keeps the longest prefix of ``base_tokens``
        that survives the truncation described by
        ``delta.removed_tail_count`` and appends
        ``delta.appended_tokens``.

        Args:
            base_tokens: Original token sequence.
            delta: Delta to apply.

        Returns:
            tuple[int, ...]: The reconstructed token sequence.

        Raises:
            ValueError: If ``removed_tail_count`` is larger than
                ``len(base_tokens)`` (the delta is malformed).
        """
        removed = delta.removed_tail_count
        if removed < 0 or removed > len(base_tokens):
            raise ValueError(
                f"Invalid removed_tail_count {removed} for base of length {len(base_tokens)}"
            )
        kept = base_tokens[: len(base_tokens) - removed]
        return kept + delta.appended_tokens
