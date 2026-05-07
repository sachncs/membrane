"""Tests for value_density module."""

import pytest

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature
from membrane.value_density import ValueDensity


def make_fragment(content_hash="abc", reuse_score=0.5):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0,),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 1)
        ),
        size=10,
        ttl=3600.0,
        reuse_score=reuse_score,
        version_id=1,
    )


class TestValueDensity:
    """Test suite for ValueDensity."""

    def test_compute_with_empty_history(self):
        vd = ValueDensity()
        frag = make_fragment(reuse_score=0.6)
        assert vd.compute(frag, []) == pytest.approx(0.6)

    def test_compute_with_access_history(self):
        vd = ValueDensity()
        frag = make_fragment(content_hash="h", reuse_score=0.3)
        history = ["h", "h"]
        score = vd.compute(frag, history)
        assert score == pytest.approx(0.3 + 2 * 0.05 + 0.1)

    def test_compute_capped_at_one(self):
        vd = ValueDensity()
        frag = make_fragment(content_hash="h", reuse_score=0.9)
        history = ["h"] * 10
        score = vd.compute(frag, history)
        assert score == 1.0

    def test_importance_multiplier(self):
        vd = ValueDensity()
        frag = make_fragment(reuse_score=0.5)
        assert vd.compute(frag, [], importance=2.0) == 1.0

    def test_recency_bonus(self):
        vd = ValueDensity()
        frag = make_fragment(content_hash="h", reuse_score=0.4)
        history = ["x", "h"]
        score = vd.compute(frag, history)
        assert score == pytest.approx(0.4 + 1 * 0.05 + 0.1)
