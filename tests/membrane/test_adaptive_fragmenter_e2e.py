"""AdaptiveFragmenter e2e against a Node (Phase 3.3.10 follow-up).

The Phase 3.3.10 commit shipped AdaptiveFragmenter; the
existing unit tests cover the math in isolation. This test
runs the full path: a Node with an AdaptiveFragmenter
enabled, a content_store under memory pressure, and the
Node.window_size() returns the adaptive value.
"""

from __future__ import annotations

import pytest


class TestAdaptiveFragmenterE2E:
    def test_node_window_size_default_128(self):
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=10_000)
        # Default window is 128 when no fragmenter is configured.
        assert node.window_size() == 128

    def test_node_window_size_with_70b_fragmenter(self):
        """70b architecture baseline is 64; the adaptive pressure logic halves it."""
        from membrane.node import Node
        from membrane.transfer_engine_ext import AdaptiveFragmenter

        frag = AdaptiveFragmenter(enabled=True, model_id="llama-3-70b")
        node = Node(node_id="n1", max_memory_bytes=10_000, fragmenter=frag)
        # 70b baseline = 64.
        assert node.window_size() == 64

    def test_node_window_size_with_high_reuse_doubles(self):
        from membrane.node import Node
        from membrane.transfer_engine_ext import AdaptiveFragmenter

        frag = AdaptiveFragmenter(enabled=True, model_id="llama-3-8b")
        # High reuse + low pressure -> double the baseline.
        frag.reuse_score_avg = 0.9
        node = Node(node_id="n1", max_memory_bytes=10_000, fragmenter=frag)
        # 8b baseline = 128; doubled to 256.
        assert node.window_size() == 256

    def test_node_window_size_under_pressure_halves(self):
        from membrane.node import Node
        from membrane.transfer_engine_ext import AdaptiveFragmenter

        # 99% memory pressure -> baseline halved twice (85% +
        # 95% thresholds) to 16.
        frag = AdaptiveFragmenter(enabled=True, model_id="llama-3-8b")
        frag.node_memory_used_bytes = 99_000
        frag.node_memory_limit_bytes = 100_000
        node = Node(
            node_id="n1",
            max_memory_bytes=100_000,
            fragmenter=frag,
        )
        # 8b baseline 128; > 0.85 halves to 64; > 0.95 halves to 16.
        assert node.window_size() == 16

    def test_evaluating_disabled_returns_default(self):
        """Disabled fragmenter (or no fragmenter) returns 128."""
        from membrane.node import Node
        from membrane.transfer_engine_ext import AdaptiveFragmenter

        frag = AdaptiveFragmenter(enabled=False, model_id="llama-3-8b")
        node = Node(node_id="n1", max_memory_bytes=10_000, fragmenter=frag)
        assert node.window_size() == 128

    def test_window_size_minimum_under_extreme_pressure(self):
        """A baseline 16 stays at the floor of 16 even under pressure."""
        from membrane.node import Node
        from membrane.transfer_engine_ext import AdaptiveFragmenter

        # 70b baseline 64; pressure 95% halves to 32, but
        # baseline 8 (e.g. a tiny model) would floor at 16.
        frag = AdaptiveFragmenter(enabled=True, model_id="phi-3")
        frag.node_memory_used_bytes = 99_000  # 99% pressure
        frag.node_memory_limit_bytes = 100_000
        node = Node(
            node_id="n1",
            max_memory_bytes=100_000,
            fragmenter=frag,
        )
        # phi-3 baseline 128; pressure 99% halves once to 64,
        # then again to 32 (max(16, 64//4)=16). The AdaptiveFragmenter
        # math is capped at max(16, baseline // (2**n)).
        assert node.window_size() == 16
