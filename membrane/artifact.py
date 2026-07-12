"""Artifact: retrieved document or embedding as a memory object.

This module defines :class:`Artifact`, Membrane's addressable
representation of a *retrieval-augmented* memory unit — typically a
document chunk or a precomputed embedding that has been pulled from
an external source (web fetch, RAG corpus, file ingestion).

Artifacts are conceptually different from
:class:`~membrane.prefix.Prefix` and
:class:`~membrane.kv_segment.KVSegment`:

* A prefix is a *logical* token sequence produced by a tokenizer.
* A KV segment is the *physical* KV-cache tensor for a model.
* An artifact is opaque content (text or embedding) associated with
  a ``source_url`` that is reused across RAG-style requests.

Because artifacts carry the original source URL, they enable
provenance tracking and (when supported by the persistence layer)
re-fetching the canonical version of the underlying content.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


@dataclass(frozen=True)
class Artifact:
    """A retrieved document or embedding artifact.

    An artifact wraps an opaque piece of retrieved content together
    with the metadata required to store, deduplicate, and reason
    about it. It is the recommended memory type for RAG-style
    workloads where the same document is referenced by many prompts.

    Attributes:
        source_url: Stable identifier or URL of the underlying
            source. May be empty when the artifact was produced
            locally without an external reference. Used for
            provenance and for cache busting when the source
            changes.
        text_hash: Hash of the artifact's raw text content. Used to
            detect updates from the upstream source.
        embedding: Dense semantic embedding produced by an embedding
            model. Stored on the fragment to enable semantic
            similarity lookups.
        content_hash: Deterministic hash of the artifact's payload
            (typically ``hash(source_url + text_hash + embedding)``).
        semantic_hash: Approximate hash used by the semantic index.
        size_bytes: Serialized storage size in bytes.
        token_count: Token count of the embedded text; ``0`` when
            the artifact carries only an embedding.
        reuse_score: Producer-supplied reuse likelihood in
            ``[0, 1]``.

    Example:
        >>> art = Artifact(
        ...     source_url="https://example.com/docs/intro",
        ...     text_hash="h1",
        ...     embedding=(0.0, 0.1, 0.2),
        ...     content_hash="c1",
        ...     semantic_hash="s1",
        ...     size_bytes=2048,
        ...     token_count=512,
        ...     reuse_score=0.7,
        ... )
        >>> frag = art.materialize()
    """

    source_url: str
    text_hash: str
    embedding: tuple[float, ...]
    content_hash: str
    semantic_hash: str
    size_bytes: int
    token_count: int
    reuse_score: float

    def materialize(self) -> Fragment:
        """Materialize this artifact into a storable :class:`Fragment`.

        The returned fragment uses ``model_id="artifact"`` so that
        indexes and routes can distinguish artifacts from
        prefix/segment-level fragments when computing placement
        decisions.

        Returns:
            Fragment: An immutable fragment carrying the artifact's
            identity, embedding, and lifecycle metadata. The
            embedding is preserved as-is (not re-generated from
            shape, unlike :meth:`~membrane.kv_segment.KVSegment
            .materialize`).
        """
        signature = StructuralSignature(
            model_id="artifact",
            layer_range=(0, 0),
            token_span=(0, self.token_count - 1),
        )
        return Fragment(
            content_hash=self.content_hash,
            embedding=self.embedding,
            structural_signature=signature,
            size=self.size_bytes,
            ttl=3600.0,
            reuse_score=self.reuse_score,
            version_id=1,
        )

    @classmethod
    def from_fragment(cls, fragment: Fragment) -> "Artifact":
        """Reconstruct an :class:`Artifact` from a stored :class:`Fragment`.

        ``source_url`` is not stored on a fragment and is recovered
        as an empty string. Callers that need the original URL must
        persist it externally (e.g., in the canonical store's
        metadata side-table).

        Args:
            fragment: A fragment previously produced by
                :meth:`materialize` (or with a structurally
                compatible signature).

        Returns:
            Artifact: An artifact with preserved ``embedding``,
            ``content_hash``, ``size_bytes``, ``token_count``,
            ``reuse_score``, and an empty ``source_url``.
        """
        span = fragment.structural_signature.token_span
        # token_span is inclusive on both ends; convert to count.
        count = span[1] - span[0] + 1
        return cls(
            # source_url cannot be reconstructed from the fragment;
            # callers needing the URL must persist it out-of-band.
            source_url="",
            text_hash=fragment.content_hash,
            embedding=fragment.embedding,
            content_hash=fragment.content_hash,
            # semantic_hash is not preserved on Fragment; reuse the
            # content_hash as a stable surrogate.
            semantic_hash=fragment.content_hash,
            size_bytes=fragment.size,
            token_count=count,
            reuse_score=fragment.reuse_score,
        )
