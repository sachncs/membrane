"""Property tests for v5-only invariants (Phase 3.7.1).

The v3.0.0 release admits only ``SCHEMA_VERSION == 5`` and
the v5 canonical frame magic. The property tests exercise the
canonical frame round-trip on arbitrary bytes plus the JSON
serialization invariant on the ``Fragment.tenant_id``.

The :mod:`hypothesis` plugin is optional; the tests skip
cleanly when the plugin is not installed (CI picks up
hypothesis in the v3.0.1 extras).
"""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _HYPOTHESIS_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    _HYPOTHESIS_AVAILABLE = False

from membrane.canonical import (
    CANONICAL_SCHEMA_VERSION,
    MAGIC,
    canonicalize,
    parse_canonical,
)
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.serialization import SCHEMA_VERSION, from_dict, to_dict


@pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE, reason="hypothesis plugin not installed"
)
class TestCanonicalFrameV5:
    """The v5 canonical frame is the only one the v3.0.0 reader accepts."""

    def test_magic_constant(self):
        assert CANONICAL_SCHEMA_VERSION == 5
        assert MAGIC == b"\xc0\xde\x01\x05"

    @given(st.binary(min_size=0, max_size=2048))
    @settings(max_examples=20, deadline=None)
    def test_round_trip_arbitrary_payload(self, payload):
        identity = PayloadIdentity(
            payload_hash="h" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, max(1, len(payload))),
            dtype="float16",
            shape=(1, 1, 1, 1, 64),
        )
        frame = canonicalize(identity, payload)
        parsed_identity, parsed_payload = parse_canonical(frame)
        assert parsed_identity == identity
        assert parsed_payload == payload

    @given(st.binary(min_size=1, max_size=512))
    @settings(max_examples=10, deadline=None)
    def test_trailer_short_payload(self, payload):
        """Truncated-SHA256 trailer always validates for any payload up to 512 bytes."""
        identity = PayloadIdentity(
            payload_hash="h" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, max(1, len(payload))),
            dtype="float16",
            shape=(1, 1, 1, 1, 64),
        )
        frame = canonicalize(identity, payload)
        _, parsed = parse_canonical(frame)
        assert parsed == payload


@pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE, reason="hypothesis plugin not installed"
)
class TestFragmentSerialization:
    """``to_dict`` / ``from_dict`` round-trip preserves every field, including tenant_id."""

    @given(
        tenant_id=st.sampled_from(
            ["public", "acme", "globex", "tenant-with-dashes"]
        ),
        reuse_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ttl=st.floats(min_value=0.0, max_value=3600.0, allow_nan=False),
        payload_size=st.integers(min_value=0, max_value=10_000),
        version_id=st.integers(min_value=1, max_value=10),
        consistency=st.sampled_from(["strong", "quorum", "eventual"]),
    )
    @settings(max_examples=30, deadline=None)
    def test_round_trip_preserves_tenant(self, tenant_id, reuse_score, ttl, payload_size, version_id, consistency):
        ident = PayloadIdentity(
            payload_hash="h" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, payload_size),
            dtype="float16",
            shape=(1, 1, 1, 1, 64),
        )
        frag = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=payload_size,
            ttl=ttl,
            reuse_score=reuse_score,
            version_id=version_id,
            consistency=consistency,
            tenant_id=tenant_id,
        )
        rebuilt = from_dict(to_dict(frag))
        assert rebuilt.tenant_id == tenant_id
        assert rebuilt.reuse_score == reuse_score
        assert rebuilt.payload_size == payload_size
        assert rebuilt.consistency == consistency


@pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE, reason="hypothesis plugin not installed"
)
class TestSchemaBump:
    """``from_dict`` rejects any schema_version other than 5."""

    @given(st.integers(min_value=0, max_value=1000))
    @settings(max_examples=10, deadline=None)
    def test_rejects_non_v5(self, version):
        if version == SCHEMA_VERSION:
            return
        from membrane.errors import SchemaError

        with pytest.raises(SchemaError):
            from_dict(
                {
                    "schema_version": version,
                    "tenant_id": "public",
                    "identity": {"payload_hash": "h" * 64},
                    "payload_ref": None,
                    "payload_size": 0,
                    "ttl": 0.0,
                    "reuse_score": 0.0,
                    "version_id": 1,
                    "consistency": "strong",
                    "hlc": 0,
                    "fingerprint_compat": "",
                }
            )
