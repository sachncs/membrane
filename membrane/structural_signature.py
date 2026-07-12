"""StructuralSignature for fragment addressing.

This module defines :class:`StructuralSignature`, an immutable descriptor
that locates a fragment within the *structural* space of a model
computation graph — i.e., which model produced it, which transformer
layers it spans, and which token positions it covers.

Unlike ``content_hash`` (which identifies *what* the bytes are),
``StructuralSignature`` identifies *where* in the computation the bytes
belong. The two are complementary: ``content_hash`` drives
deduplication and canonical storage; ``StructuralSignature`` drives
routing, position-aware caching, and graph-level reconstruction.

Use cases:
    * Position-aware placement: A prefill node can request fragments
      for layers ``[10, 20)`` only.
    * Model compatibility checks: Two fragments are interchangeable
      only if their ``model_id`` and ``schema_version`` match.
    * Graph traversal:
      :class:`membrane.fragment_graph.FragmentGraph` uses signatures as
      node identifiers when reconstructing contexts.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralSignature:
    """Immutable structural signature of a fragment.

    The signature describes a fragment's *position* in the model
    computation graph rather than its byte content. This is the key
    used by routing logic to decide whether two fragments can be
    concatenated, whether a cache hit is valid, and which layer
    boundaries must be respected during reconstruction.

    Instances are frozen and therefore hashable, which allows them to
    be used as dictionary keys in graph and index structures.

    Attributes:
        model_id: Stable identifier of the model that produced (or can
            consume) this fragment, e.g., ``"llama-3-8b"`` or
            ``"mistral-7b-instruct"``. Two fragments are only
            interchangeable for inference when their ``model_id``
            matches exactly.
        layer_range: Inclusive ``(start, end)`` range of transformer
            layer indices covered by the fragment. ``start == end``
            denotes a single layer; ``start < end`` denotes a span.
            Must satisfy ``start <= end`` and both bounds must be
            non-negative.
        token_span: Inclusive ``(start, end)`` range of token positions
            within the prompt that this fragment corresponds to. Used by
            positional indexes for prefix matching and by the
            reconstruction engine to assemble token-aligned contexts.

    Example:
        >>> from membrane.structural_signature import StructuralSignature
        >>> sig = StructuralSignature(
        ...     model_id="llama-3-8b",
        ...     layer_range=(0, 32),
        ...     token_span=(0, 128),
        ... )
        >>> sig.model_id
        'llama-3-8b'
    """

    model_id: str
    layer_range: tuple[int, int]
    token_span: tuple[int, int]
