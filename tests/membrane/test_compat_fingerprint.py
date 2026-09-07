"""Tests for ModelCompatibilityFingerprint + compute_config_hash (Phase 1.1)."""

from __future__ import annotations

import pytest

from membrane.compat import (
    ModelCompatibilityFingerprint,
    compat_hash,
    compute_config_hash,
)


class TestCompatHash:
    def test_default_fingerprint_uses_model_id_for_tokenizer(self):
        fp = compat_hash(model_id="gpt2")
        assert fp.tokenizer_name == "gpt2"
        assert fp.model_id == "gpt2"

    def test_explicit_tokenizer_kept(self):
        fp = compat_hash(model_id="gpt2", tokenizer_name="custom-tok")
        assert fp.tokenizer_name == "custom-tok"

    def test_to_dict_round_trip(self):
        fp = compat_hash(
            model_id="llama-3-8b",
            model_revision="abc123",
            model_layout_version=4,
            tokenizer_name="llama-3-8b-tokenizer",
            tokenizer_revision="def456",
            tokenizer_layout_version=2,
            dtype="bfloat16",
            config_hash="deadbeef",
        )
        round_trip = ModelCompatibilityFingerprint.from_dict(fp.to_dict())
        assert round_trip == fp

    def test_compatibility_hash_is_deterministic(self):
        fp = compat_hash(model_id="x", model_revision="r1", config_hash="c1")
        assert fp.compatibility_hash() == fp.compatibility_hash()
        # Different config_hash produces a different digest.
        fp2 = compat_hash(model_id="x", model_revision="r1", config_hash="c2")
        assert fp2.compatibility_hash() != fp.compatibility_hash()

    def test_rejects_empty_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            ModelCompatibilityFingerprint(
                model_id="",
                model_revision="",
                model_layout_version=0,
                tokenizer_name="",
                tokenizer_revision="",
                tokenizer_layout_version=0,
                dtype="float16",
                config_hash="x",
            )

    def test_rejects_negative_layout_version(self):
        with pytest.raises(ValueError, match="layout versions"):
            ModelCompatibilityFingerprint(
                model_id="x",
                model_revision="",
                model_layout_version=-1,
                tokenizer_name="x",
                tokenizer_revision="",
                tokenizer_layout_version=0,
                dtype="float16",
                config_hash="x",
            )


class TestComputeConfigHash:
    def test_dict_round_trip(self):
        config = {"hidden_size": 4096, "num_attention_heads": 32}
        h1 = compute_config_hash(config)
        h2 = compute_config_hash(dict(config))
        assert h1 == h2
        assert len(h1) == 64

    def test_dict_reorder_does_not_change_hash(self):
        # The canonical encoding sorts keys, so a dict with
        # different insertion order produces the same hash.
        a = compute_config_hash({"a": 1, "b": 2, "c": 3})
        b = compute_config_hash({"c": 3, "a": 1, "b": 2})
        assert a == b

    def test_dict_with_object_to_dict(self):
        class _Cfg:
            def to_dict(self):
                return {"a": 1, "b": 2}

        assert compute_config_hash(_Cfg()) == compute_config_hash({"a": 1, "b": 2})

    def test_rejects_object_without_to_dict(self):
        class _Cfg:
            pass

        with pytest.raises(TypeError, match="to_dict"):
            compute_config_hash(_Cfg())
