"""Tests for the v3.0.0 tenant field + schema 5 bump (Phase 3.1.5)."""

from __future__ import annotations

import pytest

from membrane.canonical import CANONICAL_SCHEMA_VERSION, MAGIC, canonicalize, parse_canonical
from membrane.errors import SchemaError
from membrane.fragment import Fragment, _validate_tenant_id
from membrane.identity import PayloadIdentity
from membrane.serialization import SCHEMA_VERSION, from_dict, to_dict


def _identity() -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash="a" * 64,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 7),
        dtype="float16",
        shape=(1, 1, 1, 8, 64),
    )


class TestSchemaBump:
    def test_schema_version_is_5(self):
        """The serialization schema is now 5; older schemas are rejected."""
        assert SCHEMA_VERSION == 5

    def test_canonical_magic_is_v5(self):
        assert MAGIC == b"\xc0\xde\x01\x05"
        assert CANONICAL_SCHEMA_VERSION == 5

    def test_v5_round_trip(self):
        buf = canonicalize(_identity(), b"hello")
        _identity_obj, payload = parse_canonical(buf)
        assert payload == b"hello"

    def test_v4_payload_rejected(self):
        """A v4 frame's magic is rejected (no compat shim)."""
        buf = canonicalize(_identity(), b"hello")
        # Mutate the magic from v5 to v4 to simulate an old frame.
        bad = b"\xc0\xde\x01\x04" + buf[4:]
        with pytest.raises(SchemaError, match="bad magic"):
            parse_canonical(bad)

    def test_v2_payload_rejected(self):
        """A v2 frame's magic is rejected (no compat shim)."""
        buf = canonicalize(_identity(), b"hello")
        bad = b"\xc0\xde\x01\x02" + buf[4:]
        with pytest.raises(SchemaError, match="bad magic"):
            parse_canonical(bad)

    def test_v4_dict_payload_rejected(self):
        """A v4 schema_version wire dict is rejected by from_dict."""
        ident = _identity()
        frag = Fragment(identity=ident, payload_ref=None, payload_size=0, ttl=60, reuse_score=0.5, version_id=1)
        wire = to_dict(frag)
        wire["schema_version"] = 4
        with pytest.raises(SchemaError, match="incompatible schema_version=4"):
            from_dict(wire)

    def test_v3_dict_payload_rejected(self):
        ident = _identity()
        frag = Fragment(identity=ident, payload_ref=None, payload_size=0, ttl=60, reuse_score=0.5, version_id=1)
        wire = to_dict(frag)
        wire["schema_version"] = 3
        with pytest.raises(SchemaError, match="incompatible schema_version=3"):
            from_dict(wire)


class TestFragmentTenantId:
    def test_default_tenant_is_public(self):
        ident = _identity()
        frag = Fragment(identity=ident, payload_ref=None, payload_size=0, ttl=60, reuse_score=0.5, version_id=1)
        assert frag.tenant_id == "public"

    def test_explicit_tenant_kept(self):
        ident = _identity()
        frag = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=0,
            ttl=60,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        assert frag.tenant_id == "acme"

    def test_empty_tenant_rejected(self):
        with pytest.raises(ValueError, match="tenant_id"):
            _validate_tenant_id("")

    def test_long_tenant_rejected(self):
        with pytest.raises(ValueError, match="tenant_id"):
            _validate_tenant_id("t" * 129)

    @pytest.mark.parametrize(
        "ch",
        [" ", ":", "/", "\\", '"', "\n", "\t", "\r"],
    )
    def test_forbidden_char_in_tenant_rejected(self, ch):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_tenant_id(f"acme{ch}co")

    def test_with_tenant_returns_new_instance(self):
        ident = _identity()
        frag = Fragment(identity=ident, payload_ref=None, payload_size=0, ttl=60, reuse_score=0.5, version_id=1)
        new = frag.with_tenant("acme")
        assert new.tenant_id == "acme"
        assert frag.tenant_id == "public"
        assert new.identity == frag.identity

    def test_with_tenant_validates(self):
        ident = _identity()
        frag = Fragment(identity=ident, payload_ref=None, payload_size=0, ttl=60, reuse_score=0.5, version_id=1)
        with pytest.raises(ValueError):
            frag.with_tenant("acme corp")  # space is forbidden

    def test_merge_across_tenants_rejected(self):
        ident = _identity()
        a = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=0,
            ttl=60,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
            hlc=1,
        )
        b = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=0,
            ttl=60,
            reuse_score=0.5,
            version_id=1,
            tenant_id="globex",
            hlc=2,
        )
        with pytest.raises(ValueError, match="across tenants"):
            a.merge(b)

    def test_merge_within_same_tenant_succeeds(self):
        ident = _identity()
        a = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=0,
            ttl=60,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
            hlc=1,
        )
        b = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=0,
            ttl=60,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
            hlc=2,
        )
        merged = a.merge(b)
        assert merged.tenant_id == "acme"
        assert merged.hlc == 2

    def test_tenant_round_trips_through_serialization(self):
        ident = _identity()
        frag = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=0,
            ttl=60,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        wire = to_dict(frag)
        assert wire["tenant_id"] == "acme"
        assert wire["schema_version"] == 5
        rebuilt = from_dict(wire)
        assert rebuilt.tenant_id == "acme"

    def test_tenant_round_trip_through_canonical_frame(self):
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity

        ident = PayloadIdentity(
            payload_hash="b" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, 7),
            dtype="float16",
            shape=(1, 1, 1, 8, 64),
        )
        buf = canonicalize(ident, b"hello")
        rebuilt_identity, payload = parse_canonical(buf)
        assert rebuilt_identity == ident
        assert payload == b"hello"
