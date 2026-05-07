"""Tests for Artifact memory object."""

from membrane.artifact import Artifact
from membrane.semantic_hash import compute_semantic_hash


def test_artifact_creation():
    a = Artifact(
        source_url="https://example.com/doc",
        text_hash="th123",
        embedding=(0.1, 0.2, 0.3),
        content_hash="ch123",
        semantic_hash=compute_semantic_hash((0.1, 0.2, 0.3)),
        size_bytes=256,
        token_count=50,
        reuse_score=0.7,
    )
    assert a.source_url == "https://example.com/doc"
    assert a.token_count == 50


def test_artifact_materialize():
    a = Artifact(
        source_url="",
        text_hash="th",
        embedding=(0.5, -0.5),
        content_hash="ch",
        semantic_hash="sh",
        size_bytes=128,
        token_count=10,
        reuse_score=0.5,
    )
    frag = a.materialize()
    assert frag.content_hash == "ch"
    assert frag.embedding == (0.5, -0.5)


def test_artifact_from_fragment():
    a = Artifact(
        source_url="src",
        text_hash="th",
        embedding=(0.1,),
        content_hash="ch",
        semantic_hash="sh",
        size_bytes=64,
        token_count=5,
        reuse_score=0.4,
    )
    frag = a.materialize()
    recon = Artifact.from_fragment(frag)
    assert recon.content_hash == "ch"
    assert recon.token_count == 5
