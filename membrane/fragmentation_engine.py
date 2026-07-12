"""FragmentationEngine: fixed-size windows, split, and merge.

This module provides the *mechanics* for turning a token sequence into
content-addressable fragments and for reshaping fragments at runtime
when prompts grow or shrink.

The module exposes:

* :class:`FragmentationConfig` — knobs controlling window size,
  embedding dimensionality, and merge thresholds.
* :func:`compute_content_hash` — deterministic MD5 hash over a token
  tuple, used as the canonical ``content_hash`` for fragments.
* :func:`generate_embedding` — synthetic normalized embedding
  derived deterministically from the token sequence. Used for
  semantic indexing without paying for a real embedding model.
* :class:`FragmentationEngine` — the high-level façade with three
  operations:

  - :meth:`FragmentationEngine.create_windows` — split a prompt into
    fixed-size fragments.
  - :meth:`FragmentationEngine.split` — subdivide a fragment at given
    token positions.
  - :meth:`FragmentationEngine.merge` — combine adjacent fragments
    into one.

Design rationale:
    Fragmentation is intentionally content-based: identical token
    sequences always produce the same ``content_hash`` and
    ``embedding``, regardless of how they were split. This is what
    makes the deduplication, semantic indexing, and cross-node
    gossip protocols work without coordination.

    ``generate_embedding`` uses a deterministic pseudo-random number
    generator seeded with the content hash. This keeps the
    semantic-hash bucket assignment stable across processes while
    avoiding the cost of running a real embedding model.

Limitations:
    * Embeddings are *not* semantically meaningful — they are
      reproducible surrogates. Replace with a real encoder for
      production similarity search.
    * :meth:`merge` refuses to combine fragments whose average
      ``reuse_score`` exceeds ``merge_reuse_threshold``. The intent
      is to keep "hot" fragments small enough to be replicated
      independently. Tune the threshold for your workload.
"""

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
        window_size: Number of tokens per initial window created by
            :meth:`FragmentationEngine.create_windows`. Larger
            windows amortize metadata overhead but reduce
            granularity for partial-prefix reuse.
        embedding_dim: Dimensionality of synthetic embeddings
            produced by :func:`generate_embedding`. Must be
            consistent across the cluster to keep semantic indexes
            interoperable.
        merge_reuse_threshold: Maximum average ``reuse_score`` for
            which :meth:`FragmentationEngine.merge` is willing to
            combine adjacent fragments. High-reuse fragments are
            kept separate so they can be replicated independently.
    """

    window_size: int = 1024
    embedding_dim: int = 128
    merge_reuse_threshold: float = 0.8


def compute_content_hash(tokens: tuple[int, ...]) -> str:
    """Compute a deterministic content hash for a token sequence.

    The function uses MD5 over the canonical string representation
    of the tuple. MD5 is used because it is fast and we do not
    require cryptographic collision resistance here — the input is
    already a sequence of integers, so attacker-controlled collisions
    are not a meaningful threat.

    Args:
        tokens: Token IDs as an immutable tuple.

    Returns:
        str: Hexadecimal MD5 digest string.
    """
    payload = str(tokens).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def generate_embedding(tokens: tuple[int, ...], dim: int) -> tuple[float, ...]:
    """Generate a deterministic normalized embedding from a token sequence.

    A linear congruential generator seeded with the content hash
    produces ``dim`` pseudo-random values in ``[-1, 1]``. The
    resulting vector is then L2-normalized so that downstream cosine
    similarities are well-defined. Two token sequences that differ
    in any token get a completely different embedding because the
    seed (the content hash) changes.

    Args:
        tokens: Token IDs as an immutable tuple.
        dim: Target embedding dimensionality. Must be positive.

    Returns:
        tuple[float, ...]: A unit-vector embedding of length
        ``dim``. Returns the zero vector when ``dim`` is zero or
        the generator produces a degenerate zero norm (which never
        happens in practice but is guarded against).
    """
    seed = int(compute_content_hash(tokens), 16)
    values = []
    for _ in range(dim):
        # Linear congruential generator with classic parameters
        # (Park-Miller variant). Deterministic and cheap.
        seed = (seed * 9301 + 49297) % 233280
        value = (seed / 233280.0) * 2.0 - 1.0
        values.append(value)
    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0.0:
        # Normalize to unit length so semantic distance metrics
        # based on dot product are well-defined.
        values = [v / norm for v in values]
    return tuple(values)


class FragmentationEngine:
    """Splits prompts into fixed-size fragments and supports split/merge.

    The engine is stateless beyond its :class:`FragmentationConfig`.
    All methods are pure functions of their inputs and the config,
    so a single instance can be shared across threads safely.
    """

    def __init__(self, config: FragmentationConfig | None = None) -> None:
        """Initialize the engine with the supplied configuration.

        Args:
            config: Fragmentation parameters. When ``None``, a
                default :class:`FragmentationConfig` is used with
                ``window_size=1024``, ``embedding_dim=128``, and
                ``merge_reuse_threshold=0.8``.
        """
        self.config = config or FragmentationConfig()
        logger.info("Initialized %s", self.__class__.__name__)

    def create_windows(
        self,
        prompt_tokens: list[int],
        model_id: str,
    ) -> list[Fragment]:
        """Create fixed-size fragment windows from a prompt.

        The prompt is split into ``ceil(len / window_size)``
        contiguous, non-overlapping fragments. Each fragment's
        ``content_hash`` is computed from its actual token slice,
        so identical prompts always yield identical fragments.

        Args:
            prompt_tokens: Token IDs to fragment. May be empty, in
                which case an empty list is returned.
            model_id: Model identifier stamped on every fragment's
                :class:`~membrane.structural_signature.StructuralSignature`.

        Returns:
            list[Fragment]: Ordered fragments covering the full
            prompt. The final fragment may be shorter than
            ``window_size`` if the prompt length is not an exact
            multiple.
        """
        if not prompt_tokens:
            return []

        fragments = []
        total_tokens = len(prompt_tokens)
        window_size = self.config.window_size
        # Ceiling division so the last window covers the residual.
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
                # 64 bytes/token is a conservative upper-bound
                # estimate for serialized KV tensors; used for
                # capacity accounting, not for transport.
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

        Split points are interpreted as absolute token positions
        (matching ``StructuralSignature.token_span``). Points
        outside the fragment's span, or duplicates, are ignored.
        Each split point becomes the inclusive start of the next
        sub-fragment.

        Args:
            fragment: Source fragment to split.
            split_points: Absolute token positions where splits
                occur.
            original_tokens: Optional full token sequence from
                which the fragment was derived. When provided and
                in range, sub-fragments use the actual token slice
                for hashing and embedding; otherwise synthetic
                placeholder tokens are used, which will not match
                the exact index.

        Returns:
            list[Fragment]: Ordered sub-fragments covering the
            original span. Returns ``[fragment]`` (no-op) when
            ``split_points`` is empty or contains no valid points.
        """
        if not split_points:
            return [fragment]

        f_start, f_end = fragment.structural_signature.token_span
        # Drop points outside the span and deduplicate while
        # preserving order via sorted().
        points = sorted({p for p in split_points if f_start < p < f_end})
        if not points:
            return [fragment]

        # Build half-open boundaries [start, end) for slicing. The
        # first boundary is the fragment's start; each split point
        # becomes the start of the next chunk; the final boundary
        # is one past the fragment's last token.
        boundaries = [f_start] + [p + 1 for p in points] + [f_end + 1]
        fragments = []
        model_id = fragment.structural_signature.model_id
        layer_range = fragment.structural_signature.layer_range

        for i in range(len(boundaries) - 1):
            sub_start = boundaries[i]
            sub_end = boundaries[i + 1] - 1

            if original_tokens is not None and sub_end < len(original_tokens):
                # Use the actual token slice so the sub-fragment
                # is content-addressable against the exact index.
                sub_tokens = tuple(original_tokens[sub_start : sub_end + 1])
            else:
                # Fall back to synthetic tokens. This loses
                # content-addressability against the exact index
                # but keeps the API usable when the caller does
                # not have the original tokens handy.
                sub_tokens = tuple(range(sub_start, sub_end + 1))
                if original_tokens is not None:
                    logger.warning(
                        "Split at (%d, %d) exceeds original_tokens length %d; using synthetic tokens",
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
                # Preserve lifecycle metadata from the parent so
                # child fragments don't appear "fresher" than
                # their ancestor.
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
        """Merge adjacent fragments when they share a model and are cold.

        Merge is only attempted when:

        1. There are at least two fragments.
        2. Every fragment shares ``model_id`` and ``layer_range``
           with the first one.
        3. The fragments form a *contiguous* span (each fragment's
           ``token_span[0]`` equals the previous fragment's
           ``token_span[1] + 1``).
        4. The average ``reuse_score`` does not exceed
           ``merge_reuse_threshold``. Hot fragments are kept
           separate so they can be replicated independently.

        Args:
            fragments: Candidate fragments to merge. Must be ordered
                by ``token_span``.
            token_sequences: Optional token sequences for each
                fragment. When provided and matching the fragment
                count, the merged fragment uses the concatenated
                tokens for hashing and embedding; otherwise
                synthetic placeholder tokens are used, which will
                not match the exact index.

        Returns:
            Fragment | None: The merged fragment, or ``None`` when
            any of the merge rules above rejects the operation.
        """
        if len(fragments) < 2:
            return None

        first = fragments[0]
        model_id = first.structural_signature.model_id
        layer_range = first.structural_signature.layer_range
        total_size = 0
        total_reuse = 0.0

        for i, frag in enumerate(fragments):
            # All fragments must come from the same model and layer.
            if frag.structural_signature.model_id != model_id:
                return None
            if frag.structural_signature.layer_range != layer_range:
                return None
            # Contiguity check: each fragment must start exactly
            # one token after the previous one ended.
            if i > 0:
                prev_end = fragments[i - 1].structural_signature.token_span[1]
                curr_start = frag.structural_signature.token_span[0]
                if curr_start != prev_end + 1:
                    return None
            total_size += frag.size
            total_reuse += frag.reuse_score

        avg_reuse = total_reuse / len(fragments)
        # Refuse to merge hot fragments — keeping them separate
        # allows finer-grained replication.
        if avg_reuse > self.config.merge_reuse_threshold:
            return None

        start = first.structural_signature.token_span[0]
        end = fragments[-1].structural_signature.token_span[1]

        if token_sequences is not None and len(token_sequences) == len(fragments):
            # Concatenate the actual token sequences so the merged
            # fragment is content-addressable against the exact
            # index.
            merged_tokens_list: list[int] = []
            for seq in token_sequences:
                merged_tokens_list.extend(seq)
            merged_tokens = tuple(merged_tokens_list)
        else:
            # Fall back to synthetic tokens. The fragment will
            # not dedupe against a true exact-match index.
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
            # Inherit TTL from the earliest (lowest-version) child.
            ttl=first.ttl,
            # Use the average reuse as a smoother signal than any
            # single fragment's score.
            reuse_score=avg_reuse,
            # Version id is the max across children so we never
            # silently regress to an older revision.
            version_id=max(f.version_id for f in fragments),
        )
