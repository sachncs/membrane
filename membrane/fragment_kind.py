"""FragmentKind: taxonomy of synthetic model_ids used in Membrane.

The four memory objects (:class:`~membrane.prefix.Prefix`,
:class:`~membrane.segment.Segment`, :class:`~membrane.artifact.Artifact`,
:class:`~membrane.trace.Trace`) and the weighted graph all materialize
fragments whose :class:`~membrane.identity.PayloadIdentity.model_id`
is a short string discriminator. The strings themselves are part of
Membrane's wire format (they appear in serialized fragments and in the
proto schema indirectly via the embedding / reuse_score fields), so
they must not change.

:class:`FragmentKind` exists so the codebase has a single, type-safe
reference to those discriminator strings. Every site that synthesizes
or inspects a memory-object fragment's ``model_id`` should go through
this enum.

Thread safety:
    The enum is immutable; safe to share across threads.
"""

from __future__ import annotations

from enum import Enum


class FragmentKind(str, Enum):
    """Discriminator for Membrane memory-object fragment types.

    Values are the wire-format strings used by
    :class:`membrane.fragment.Fragment.identity.model_id`.
    The enum inherits from ``str`` so :class:`FragmentKind` is usable
    anywhere a string is expected, including equality with raw
    ``model_id`` strings stored on legacy data.

    Attributes:
        PREFIX: Public / common prefixes that any node may share.
        KV: Per-layer/head KV slice, model-specific.
        ARTIFACT: Retrieved document (RAG-style).
        TRACE: Tool invocation output.
        WEIGHTED: Synthesized by the weighted-graph layer for
            placeholder edges.
    """

    PREFIX = "prefix"
    KV = "kv"
    ARTIFACT = "artifact"
    TRACE = "tool"
    WEIGHTED = "weighted_graph"


__all__ = ["FragmentKind"]
