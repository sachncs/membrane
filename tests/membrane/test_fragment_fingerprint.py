"""Tests for Fragment.fingerprint_compat + MembraneValidator (Phase 1.3)."""

from __future__ import annotations

import pytest

from membrane.compat import (
    MembraneIncompatibleError,
    MembraneValidator,
    ModelCompatibilityFingerprint,
    compat_hash,
)
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


def _identity() -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash="a" * 64,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 10),
        dtype="float16",
        shape=(1, 1, 1, 11, 64),
    )


def _fragment(fingerprint_compat: str = "") -> Fragment:
    return Fragment(
        identity=_identity(),
        payload_ref="blob",
        payload_size=10,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
        fingerprint_compat=fingerprint_compat,
    )


class TestFragmentFingerprintField:
    def test_default_empty(self):
        assert _fragment().fingerprint_compat == ""

    def test_round_trip_via_to_wire_dict(self):
        from membrane.serialization import from_dict, to_dict

        fp = compat_hash(model_id="m", config_hash="abc").compatibility_hash()
        frag = _fragment(fingerprint_compat=fp)
        round_trip = from_dict(to_dict(frag))
        assert round_trip.fingerprint_compat == fp

    def test_validates_64_char_hex(self):
        # Too short.
        with pytest.raises(ValueError, match="64-char hex"):
            _fragment(fingerprint_compat="abcd")
        # Non-hex characters.
        with pytest.raises(ValueError, match="64-char hex"):
            _fragment(fingerprint_compat="g" * 64)
        # Empty string is the documented escape hatch.
        frag = _fragment(fingerprint_compat="")
        assert frag.fingerprint_compat == ""


class TestMembraneValidator:
    def test_validator_accepts_matching_fingerprint(self):
        fp = compat_hash(model_id="m", config_hash="abc")
        frag = _fragment(fingerprint_compat=fp.compatibility_hash())
        validator = MembraneValidator(fp)
        validator.validate(frag)  # no exception

    def test_validator_rejects_mismatch(self):
        live = compat_hash(model_id="m", config_hash="abc")
        stored_fp = compat_hash(model_id="m", config_hash="DIFFERENT")
        frag = _fragment(fingerprint_compat=stored_fp.compatibility_hash())
        validator = MembraneValidator(live)
        with pytest.raises(MembraneIncompatibleError, match="disagrees"):
            validator.validate(frag)

    def test_validator_rejects_empty_fingerprint(self):
        fp = compat_hash(model_id="m")
        frag = _fragment(fingerprint_compat="")
        validator = MembraneValidator(fp)
        with pytest.raises(MembraneIncompatibleError, match="no compatibility"):
            validator.validate(frag)
