"""Tests for the GPUDirect + adaptive fragment sizing surfaces (Phase 3.3.9-3.3.10)."""

from __future__ import annotations

import pytest

from membrane.transfer_engine_ext import (
    AdaptiveFragmenter,
    ModelSizeProfile,
    PinnedTensorHandle,
    get_model_profile,
)


class TestPinnedTensorHandle:
    def test_size_bytes(self):
        h = PinnedTensorHandle(data=b"hello", shape=(2, 4), dtype="float16")
        assert h.size_bytes() == 5


class TestModelProfile:
    def test_known_model(self):
        profile = get_model_profile("Llama-3-8B")
        assert profile.baseline_window_size == 128

    def test_unknown_model_returns_generic_profile(self):
        profile = get_model_profile("unknown-llm")
        assert profile.model_id == "unknown-llm"
        assert profile.baseline_window_size == 128


class TestAdaptiveFragmenter:
    def test_disabled_returns_default(self):
        frag = AdaptiveFragmenter(enabled=False)
        assert frag.window_size() == 128

    def test_low_memory_pressure_full_baseline(self):
        frag = AdaptiveFragmenter(enabled=True, model_id="llama-3-8b")
        assert frag.window_size() == 128

    def test_high_memory_pressure_shrinks(self):
        frag = AdaptiveFragmenter(
            enabled=True,
            model_id="llama-3-70b",
            node_memory_used_bytes=900 * 1024 * 1024,
            node_memory_limit_bytes=1024 * 1024 * 1024,
        )
        # 87% pressure, baseline 64, halved = 32.
        assert frag.window_size() == 32

    def test_high_reuse_grows_window(self):
        frag = AdaptiveFragmenter(
            enabled=True,
            model_id="llama-3-8b",
            reuse_score_avg=0.9,
        )
        # High reuse + low pressure = 2x baseline = 256.
        assert frag.window_size() == 256

    def test_combined_pressure_and_reuse(self):
        frag = AdaptiveFragmenter(
            enabled=True,
            model_id="llama-3-70b",
            reuse_score_avg=0.9,
            node_memory_used_bytes=900 * 1024 * 1024,
            node_memory_limit_bytes=1024 * 1024 * 1024,
        )
        # 70b baseline = 64; pressure halves to 32; high
        # reuse with pressure still under 0.6 is false so
        # no growth. Result: 32.
        assert frag.window_size() == 32
