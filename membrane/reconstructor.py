"""Reconstructor: rebuild context via exact/positional/semantic lookup.

This module defines :class:`Reconstructor` and the supporting
:class:`ReconstructorResult` and :class:`ReconstructorConfig`
dataclasses. The engine is responsible for assembling a fragment
chain that covers as much of a prompt as possible, falling back to
prefill computation when the cache cannot satisfy the request.

Workflow:

1. **Exact lookup** — find the longest cached prefix.
2. **Positional extension** — walk forward through adjacent
   fragments to fill in additional tokens.
3. **Semantic fill** — search the semantic index for fragments
   that cover any remaining gaps.
4. **Prefill fallback** — when a gap is too large to ignore,
   delegate to a :class:`~membrane.prefill_adapter.Adapter`
   to compute the missing tokens and index the new fragments.
5. **Post-processing** — deduplicate, sort by token span, and
   record co-access edges between the assembled fragments.

The engine is content-agnostic: it operates on whatever fragments
are present in the supplied :class:`~membrane.index
.Index`. Callers that need model-specific behavior should
select a :class:`Adapter` accordingly.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmenter import compute_content_hash, generate_embedding
from membrane.index import Index
from membrane.prefilling import Adapter


@dataclass(frozen=True)
class ReconstructorResult:
    """Outcome of a context reconstruction attempt.

    Attributes:
        fragments: Assembled fragment chain covering the
            prompt. May be empty when nothing matched.
        coverage_ratio: Fraction of prompt tokens covered by
            fragments, in ``[0, 1]``.
        missing_segments: Token ranges that still required
            prefill fallback (or were below the
            ``max_gap_tokens`` threshold and not prefilled).
        prefill_invoked: Whether prefill was needed for any gap.
    """

    fragments: list[Fragment]
    coverage_ratio: float
    missing_segments: list[tuple[int, int]]
    prefill_invoked: bool


@dataclass(frozen=True)
class ReconstructorConfig:
    """Configuration for context reconstruction thresholds.

    Attributes:
        max_gap_tokens: Maximum uncovered gap (in tokens) before
            the engine invokes prefill fallback.
        max_prefix_attempts: Maximum number of prefix lengths
            scanned when searching for an exact match. Caps the
            worst-case cost of
            :meth:`Reconstructor.longest_match`.
    """

    max_gap_tokens: int = 256
    max_prefix_attempts: int = 128


class Reconstructor:
    """Reconstructs a prompt context from fragments.

    Workflow:
        1. Exact lookup for longest prefix match.
        2. Positional extension for adjacent fragments.
        3. Semantic fill for gaps.
        4. Prefill fallback for uncovered segments.
        5. Store new fragments in the index system.

    The engine does not own its dependencies; both
    ``index_system`` and ``prefill_adapter`` are supplied by the
    caller, which makes the engine easy to swap in tests.
    """

    def __init__(
        self,
        index_system: Index,
        prefill_adapter: Adapter,
        config: ReconstructorConfig | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            index_system: In-memory indices for
                exact/semantic/positional lookup.
            prefill_adapter: Adapter for prefill fallback.
            config: Reconstruction thresholds. A default
                :class:`ReconstructorConfig` is used when
                ``None``.
        """
        self.index_system = index_system
        self.prefill_adapter = prefill_adapter
        self.config = config or ReconstructorConfig()
        logger.info("Initialized %s", self.__class__.__name__)

    def rebuild_context(
        self,
        prompt_tokens: list[int],
        model_id: str,
    ) -> ReconstructorResult:
        """Rebuild context for a prompt.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            ReconstructorResult: Assembled fragments plus
            coverage statistics and prefill bookkeeping.
        """
        if not prompt_tokens:
            # Trivial case: an empty prompt is "fully covered".
            return ReconstructorResult(
                fragments=[],
                coverage_ratio=1.0,
                missing_segments=[],
                prefill_invoked=False,
            )

        length = len(prompt_tokens)
        assembled: list[Fragment] = []
        missing_segments: list[tuple[int, int]] = []
        prefill_invoked = False
        coverage = [False] * length

        # Step 1: longest exact prefix match.
        longest = self.longest_match(prompt_tokens, model_id)
        if longest is not None:
            assembled.append(longest)
            start, end = longest.identity.token_span
            for i in range(max(0, start), min(end + 1, length)):
                coverage[i] = True

        # Step 2: extend forward via the positional index.
        current_end = self.covered_end(coverage)
        while current_end < length - 1:
            next_frag = self.find_next_adjacent(prompt_tokens, model_id, current_end)
            if next_frag is None:
                break
            assembled.append(next_frag)
            f_start, f_end = next_frag.identity.token_span
            for i in range(max(0, f_start), min(f_end + 1, length)):
                coverage[i] = True
            current_end = self.covered_end(coverage)

        # Step 3: fill gaps via the semantic index.
        gaps = self.find_gaps(coverage)
        for gap_start, gap_end in gaps:
            gap_tokens = tuple(prompt_tokens[gap_start : gap_end + 1])
            gap_embedding = generate_embedding(gap_tokens, 128)
            candidates = self.index_system.semantic_lookup(gap_embedding, k=3)
            for cand in candidates:
                c_start, c_end = cand.identity.token_span
                if c_start >= gap_start and c_end <= gap_end:
                    assembled.append(cand)
                    for i in range(c_start, min(c_end + 1, length)):
                        coverage[i] = True
                    break

        # Step 4: prefill fallback for remaining gaps.
        remaining_gaps = self.find_gaps(coverage)
        for gap_start, gap_end in remaining_gaps:
            remaining_tokens = prompt_tokens[gap_start : gap_end + 1]
            if len(remaining_tokens) > self.config.max_gap_tokens:
                # Delegate to the prefill adapter for
                # computation that the cache cannot satisfy.
                result = self.prefill_adapter.prefill(remaining_tokens, model_id)
                if result.fragments:
                    prefill_invoked = True
                    for frag in result.fragments:
                        assembled.append(frag)
                        # Index the new fragment with no
                        # locations — the local index is enough.
                        self.index_system.insert(frag, set())
                        f_start, f_end = frag.identity.token_span
                        # Map fragment span (relative to the gap)
                        # to absolute prompt positions.
                        abs_start = gap_start + f_start
                        abs_end = gap_start + f_end
                        for i in range(max(0, abs_start), min(abs_end + 1, length)):
                            coverage[i] = True
                    missing_segments.append((gap_start, gap_end))
            else:
                # Gap is small enough to leave to the decoder.
                missing_segments.append((gap_start, gap_end))

        # Step 5: deduplicate, sort, and record co-access edges.
        assembled = self.deduplicate_and_sort(assembled)
        self.record_graph_links(assembled)

        covered_count = sum(coverage)
        coverage_ratio = covered_count / length if length > 0 else 1.0

        return ReconstructorResult(
            fragments=assembled,
            coverage_ratio=coverage_ratio,
            missing_segments=missing_segments,
            prefill_invoked=prefill_invoked,
        )

    def longest_match(
        self,
        prompt_tokens: list[int],
        model_id: str,
        max_prefix_attempts: int | None = None,
    ) -> Fragment | None:
        """Find the longest prefix of ``prompt_tokens`` that exists in the exact index.

        .. warning::
            This method computes an MD5 hash for every prefix
            length tried. For a prompt of length ``L``, the
            worst-case cost is O(L^2) in token count. The
            ``max_prefix_attempts`` parameter caps the number
            of lengths scanned; if your prompts are longer than
            this cap, the method may miss an exact match that
            is shorter than the skipped lengths.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier; only fragments whose
                :class:`PayloadIdentity` carries this ``model_id``
                are considered matches.
            max_prefix_attempts: Maximum number of prefix
                lengths to try. Defaults to
                ``config.max_prefix_attempts`` (128).

        Returns:
            Fragment | None: Longest matching fragment, or
            ``None`` if no exact match exists.
        """
        if max_prefix_attempts is None:
            max_prefix_attempts = self.config.max_prefix_attempts
        length = len(prompt_tokens)
        best: Fragment | None = None
        # Scan from longest to shortest, but cap the number of
        # attempts to keep latency bounded.
        step = max(1, length // max_prefix_attempts) if max_prefix_attempts > 0 else 1
        for i in range(length, 0, -step):
            prefix = tuple(prompt_tokens[:i])
            h = compute_content_hash(prefix)
            entry = self.index_system.exact_lookup(h)
            if entry is not None:
                frag = entry.fragment
                # Only accept fragments produced by the same
                # model — different models produce
                # incompatible KV tensors.
                if frag.identity.model_id == model_id and (
                    best is None
                    or i > (best.identity.token_span[1] - best.identity.token_span[0] + 1)
                ):
                    best = frag
        return best

    def find_next_adjacent(
        self,
        prompt_tokens: list[int],
        model_id: str,
        current_end: int,
    ) -> Fragment | None:
        """Find a fragment adjacent to ``current_end`` via the positional index.

        Only candidates with the correct ``model_id`` whose span
        starts at or after ``current_end + 1`` are eligible.

        Args:
            prompt_tokens: Input token IDs (used to bound the
                candidate's span within the prompt).
            model_id: Model identifier.
            current_end: Last covered token position.

        Returns:
            Fragment | None: Adjacent fragment if found,
            otherwise ``None``.
        """
        candidates = self.index_system.positional_adjacent(current_end, max_gap=self.config.max_gap_tokens)
        for cand in candidates:
            if cand.identity.model_id != model_id:
                continue
            c_start, c_end = cand.identity.token_span
            if c_start <= current_end:
                # Not strictly after the current end — skip.
                continue
            if c_start >= 0 and c_end < len(prompt_tokens):
                return cand
        return None

    def find_gaps(self, coverage: list[bool]) -> list[tuple[int, int]]:
        """Find uncovered token ranges in the coverage bitmap.

        Args:
            coverage: Boolean list where ``True`` means covered.

        Returns:
            list[tuple[int, int]]: List of ``(start, end)``
            inclusive ranges that are uncovered, in left-to-right
            order.
        """
        gaps: list[tuple[int, int]] = []
        n = len(coverage)
        i = 0
        while i < n:
            if not coverage[i]:
                # Found the start of a gap — extend until the
                # next covered position (or end of prompt).
                start = i
                while i < n and not coverage[i]:
                    i += 1
                gaps.append((start, i - 1))
            else:
                i += 1
        return gaps

    def covered_end(self, coverage: list[bool]) -> int:
        """Return the rightmost index that is covered.

        Args:
            coverage: Boolean coverage bitmap.

        Returns:
            int: Rightmost covered index, or ``-1`` when no
            token is covered.
        """
        for i in range(len(coverage) - 1, -1, -1):
            if coverage[i]:
                return i
        return -1

    def deduplicate_and_sort(self, fragments: list[Fragment]) -> list[Fragment]:
        """Remove duplicate fragments and sort by token span start.

        Args:
            fragments: Raw assembled fragment list.

        Returns:
            list[Fragment]: Deduplicated fragments ordered by
            their token span's start position.
        """
        seen: set[str] = set()
        unique: list[Fragment] = []
        for frag in fragments:
            if frag.identity.payload_hash not in seen:
                seen.add(frag.identity.payload_hash)
                unique.append(frag)
        # Sort by token-span start so downstream consumers can
        # consume the chain in left-to-right order without
        # re-sorting.
        unique.sort(key=lambda f: f.identity.token_span[0])
        return unique

    def record_graph_links(self, fragments: list[Fragment]) -> None:
        """Record co-access edges between the assembled fragments.

        Edges are recorded for every pair so the co-access graph
        reflects the request's *full* co-occurrence. This is more
        aggressive than what a real production system would want
        (it can blow up quadratically for very large
        reconstructions), but matches the prototype's intent of
        keeping the graph dense enough to power prefetching.

        Args:
            fragments: Ordered list of fragments from a
            reconstruction.
        """
        hashes = [f.identity.payload_hash for f in fragments]
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                self.index_system.record_co_access(hashes[i], hashes[j])
