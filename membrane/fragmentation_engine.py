"""FragmentationEngine: fixed-size windows, split, and merge."""

import logging

logger = logging.getLogger(__name__)


import hashlib
import math
from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class FragmentationConfig:
    """Configuration for fragment generation.

    Attributes:
        window_size: Number of tokens per initial window.
        embedding_dim: Dimension of synthetic embedding vectors.
        merge_reuse_threshold: Max average reuse_score for merge eligibility.
    """

    window_size: int = 1024
    embedding_dim: int = 128
    merge_reuse_threshold: float = 0.8


def compute_content_hash(tokens: tuple[int, ...]) -> str:
    """Compute a deterministic content hash for a token sequence.

    Args:
        tokens: Token IDs as an immutable tuple.

    Returns:
        Hexadecimal MD5 digest string.
    """
    payload = str(tokens).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def generate_embedding(tokens: tuple[int, ...], dim: int) -> tuple[float, ...]:
    """Generate a deterministic normalized embedding from token sequence.

    Args:
        tokens: Token IDs as an immutable tuple.
        dim: Target embedding dimension.

    Returns:
        Normalized unit-vector embedding tuple.
    """
    seed = int(compute_content_hash(tokens), 16)
    values = []
    for i in range(dim):
        seed = (seed * 9301 + 49297) % 233280
        value = (seed / 233280.0) * 2.0 - 1.0
        values.append(value)
    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0.0:
        values = [v / norm for v in values]
    return tuple(values)


class FragmentationEngine:
    """Splits prompts into fixed-size fragments and supports split/merge."""

    def __init__(self, config: FragmentationConfig | None = None) -> None:
        """Initialize with optional configuration.

        Args:
            config: Fragmentation parameters. Defaults to window_size=1024.
        """
        self.config = config or FragmentationConfig()
        logger.info("Initialized %s", self.__class__.__name__)

    def create_windows(
        self,
        prompt_tokens: list[int],
        model_id: str,
    ) -> list[Fragment]:
        """Create fixed-size fragment windows from a prompt.

        Args:
            prompt_tokens: List of token IDs.
            model_id: Model identifier for structural signature.

        Returns:
            Ordered list of fragments covering the full prompt.
        """
        if not prompt_tokens:
            return []

        fragments = []
        total_tokens = len(prompt_tokens)
        window_size = self.config.window_size
        num_windows = (total_tokens + window_size - 1) // window_size

        for i in range(num_windows):
            start = i * window_size
            end = min(start + window_size, total_tokens) - 1
            window_tokens = tuple(prompt_tokens[start : end + 1])

            content_hash = compute_content_hash(window_tokens)
            embedding = generate_embedding(window_tokens, self.config.embedding_dim)
            signature = StructuralSignature(
                model_id=model_id,
                layer_range=(0, 0),
                token_span=(start, end),
            )

            frag = Fragment(
                content_hash=content_hash,
                embedding=embedding,
                structural_signature=signature,
                size=len(window_tokens) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)

        return fragments

    def split(
        self,
        fragment: Fragment,
        split_points: list[int],
        original_tokens: tuple[int, ...] | None = None,
    ) -> list[Fragment]:
        """Split a fragment at given token positions within its span.

        Args:
            fragment: Source fragment to split.
            split_points: Absolute token positions where splits occur.
            original_tokens: Optional full token sequence from which the fragment
                was derived. When provided, sub-fragments use the actual token
                slice for hashing and embedding; otherwise synthetic placeholder
                tokens are used, which will not match the exact index.

        Returns:
            Ordered sub-fragments covering the original span.
        """
        if not split_points:
            return [fragment]

        f_start, f_end = fragment.structural_signature.token_span
        points = sorted({p for p in split_points if f_start < p < f_end})
        if not points:
            return [fragment]

        boundaries = [f_start] + [p + 1 for p in points] + [f_end + 1]
        fragments = []
        model_id = fragment.structural_signature.model_id
        layer_range = fragment.structural_signature.layer_range

        for i in range(len(boundaries) - 1):
            sub_start = boundaries[i]
            sub_end = boundaries[i + 1] - 1

            if original_tokens is not None and sub_end < len(original_tokens):
                sub_tokens = tuple(original_tokens[sub_start : sub_end + 1])
            else:
                sub_tokens = tuple(range(sub_start, sub_end + 1))
                if original_tokens is not None:
                    logger.warning(
                        "Split at (%d, %d) exceeds original_tokens length %d; "
                        "using synthetic tokens",
                        sub_start,
                        sub_end,
                        len(original_tokens),
                    )

            content_hash = compute_content_hash(sub_tokens)
            embedding = generate_embedding(sub_tokens, len(fragment.embedding))
            signature = StructuralSignature(
                model_id=model_id,
                layer_range=layer_range,
                token_span=(sub_start, sub_end),
            )

            sub_frag = Fragment(
                content_hash=content_hash,
                embedding=embedding,
                structural_signature=signature,
                size=(sub_end - sub_start + 1) * 64,
                ttl=fragment.ttl,
                reuse_score=fragment.reuse_score,
                version_id=fragment.version_id,
            )
            fragments.append(sub_frag)

        return fragments

    def merge(
        self,
        fragments: list[Fragment],
        token_sequences: list[tuple[int, ...]] | None = None,
    ) -> Fragment | None:
        """Merge adjacent fragments if they share model_id and have low reuse_score.

        Args:
            fragments: Candidate fragments to merge. Must be ordered by token_span.
            token_sequences: Optional token sequences for each fragment. When
                provided, the merged fragment uses the concatenated actual tokens
                for hashing and embedding; otherwise synthetic placeholder tokens
                are used, which will not match the exact index.

        Returns:
            Merged fragment, or None if merge rules reject the operation.
        """
        if len(fragments) < 2:
            return None

        first = fragments[0]
        model_id = first.structural_signature.model_id
        layer_range = first.structural_signature.layer_range
        total_size = 0
        total_reuse = 0.0

        for i, frag in enumerate(fragments):
            if frag.structural_signature.model_id != model_id:
                return None
            if frag.structural_signature.layer_range != layer_range:
                return None
            if i > 0:
                prev_end = fragments[i - 1].structural_signature.token_span[1]
                curr_start = frag.structural_signature.token_span[0]
                if curr_start != prev_end + 1:
                    return None
            total_size += frag.size
            total_reuse += frag.reuse_score

        avg_reuse = total_reuse / len(fragments)
        if avg_reuse > self.config.merge_reuse_threshold:
            return None

        start = first.structural_signature.token_span[0]
        end = fragments[-1].structural_signature.token_span[1]

        if token_sequences is not None and len(token_sequences) == len(fragments):
            merged_tokens_list: list[int] = []
            for seq in token_sequences:
                merged_tokens_list.extend(seq)
            merged_tokens = tuple(merged_tokens_list)
        else:
            merged_tokens = tuple(range(start, end + 1))
            if token_sequences is not None:
                logger.warning(
                    "token_sequences length %d does not match fragments count %d; "
                    "using synthetic tokens for merged fragment",
                    len(token_sequences) if token_sequences else 0,
                    len(fragments),
                )

        content_hash = compute_content_hash(merged_tokens)
        embedding = generate_embedding(merged_tokens, len(first.embedding))
        signature = StructuralSignature(
            model_id=model_id,
            layer_range=layer_range,
            token_span=(start, end),
        )

        return Fragment(
            content_hash=content_hash,
            embedding=embedding,
            structural_signature=signature,
            size=total_size,
            ttl=first.ttl,
            reuse_score=avg_reuse,
            version_id=max(f.version_id for f in fragments),
        )
