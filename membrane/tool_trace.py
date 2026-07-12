"""ToolTrace: structured tool output as a memory object.

This module defines :class:`ToolTrace`, the addressable memory
representation of a structured tool invocation result. It allows
Membrane to cache and reuse tool outputs across requests — for
example, the result of a database query, an HTTP fetch, or a code
execution — in the same content-addressed fabric that caches
prefixes and KV segments.

A ``ToolTrace`` carries enough metadata to:

* Identify the originating tool (``tool_name``).
* Detect when the same input is reused (``input_hash``).
* Detect when the tool output has changed (``output_hash``).
* Reconstruct a structured representation on the receiving side
  (``structured_output``, typically a JSON string).
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import generate_embedding
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class ToolTrace:
    """A structured output from a tool invocation.

    A trace captures one (tool, input) → output mapping. Two traces
    with identical ``input_hash`` *and* ``output_hash`` are
    considered functionally equivalent and can be deduplicated by
    the canonical store.

    Attributes:
        tool_name: Name of the tool that produced the output. Used
            for routing and observability.
        input_hash: Hash of the tool's input parameters. Two tool
            calls with the same ``input_hash`` should produce the
            same output if the tool is deterministic.
        output_hash: Hash of the serialized tool output. Independent
            from ``input_hash`` to allow caching of identical outputs
            produced by different inputs (e.g., equivalent APIs).
        structured_output: JSON-serializable string of the tool
            output. May be empty when the tool produced no structured
            payload.
        content_hash: Deterministic identity hash computed from the
            canonical combination of the above fields.
        semantic_hash: Approximate hash used by the semantic index
            for similarity lookups across tool traces.
        size_bytes: Serialized payload size in bytes.
        reuse_score: Producer-supplied reuse likelihood in
            ``[0, 1]``.

    Example:
        >>> trace = ToolTrace(
        ...     tool_name="wikipedia.search",
        ...     input_hash="i1",
        ...     output_hash="o1",
        ...     structured_output='{"title": "Membrane"}',
        ...     content_hash="c1",
        ...     semantic_hash="s1",
        ...     size_bytes=64,
        ...     reuse_score=0.6,
        ... )
        >>> frag = trace.materialize()
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
        """Materialize this trace into a storable :class:`Fragment`.

        The embedding is generated from the byte values of the
        structured output so that semantically similar tool outputs
        cluster together in the semantic index. The structural
        signature uses ``model_id="tool"`` to distinguish trace
        fragments from prefix/segment/artifact fragments.

        Returns:
            Fragment: An immutable fragment carrying this trace's
            identity and embedding. The ``token_span`` is sized to
            match the length of the structured output, with a lower
            bound of ``0`` to handle empty outputs gracefully.
        """
        # Map each character of the structured output to an integer
        # token so the embedding reflects the literal content rather
        # than just the output hash.
        tokens = tuple(ord(c) for c in self.structured_output)
        embedding = generate_embedding(tokens, 128)
        # The synthetic "tool" model_id marks the fragment as a tool
        # trace, allowing downstream routing/eviction logic to treat
        # it differently from prefix or KV fragments if needed.
        signature = StructuralSignature(
            model_id="tool",
            layer_range=(0, 0),
            # Use max(0, len-1) so empty outputs produce a valid
            # (0, 0) span rather than a negative bound.
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
        """Reconstruct a :class:`ToolTrace` from a stored :class:`Fragment`.

        ``tool_name``, ``input_hash``, and ``structured_output`` are
        not preserved on a fragment. They are recovered as empty
        strings; callers needing the original values must persist
        them externally (e.g., in a side-table keyed by
        ``content_hash``).

        Args:
            fragment: A fragment previously produced by
                :meth:`materialize` (or with a structurally
                compatible signature).

        Returns:
            ToolTrace: A trace with preserved ``output_hash``,
            ``content_hash``, ``size_bytes``, ``reuse_score``, and
            empty strings for the non-serializable fields.
        """
        return cls(
            tool_name="",
            input_hash="",
            # Preserve output_hash via content_hash — the fragment
            # does not store output_hash separately, so we treat the
            # fragment's content_hash as the closest stable proxy.
            output_hash=fragment.content_hash,
            structured_output="",
            content_hash=fragment.content_hash,
            # semantic_hash is not preserved on Fragment; reuse the
            # content_hash as a stable surrogate.
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            reuse_score=fragment.reuse_score,
        )
