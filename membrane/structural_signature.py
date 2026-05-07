"""StructuralSignature for fragment addressing."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralSignature:
    """Immutable structural signature of a fragment.

    Attributes:
        model_id: Identifier of the model that produced the fragment.
        layer_range: Inclusive range of layers (start, end).
        token_span: Inclusive range of token positions (start, end).
    """

    model_id: str
    layer_range: tuple[int, int]
    token_span: tuple[int, int]
