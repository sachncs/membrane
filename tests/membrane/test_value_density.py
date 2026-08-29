from tests.conftest import make_fragment

"""Tests for density function."""

import pytest

from membrane.density import density
from membrane.fragment import Fragment
from membrane.signature import Signature


def test_compute_with_empty_history():
    """Empty history falls back to intrinsic reuse_score."""
    frag = make_fragment(reuse_score=0.6)
    assert density(frag, []) == pytest.approx(0.6)


def test_compute_with_access_history():
    """History drives both frequency and recency signals."""
    frag = make_fragment(content_hash="h", reuse_score=0.3)
    history = ["h", "h"]
    score = density(frag, history)
    assert score == pytest.approx(0.3 + 2 * 0.05 + 0.1)


def test_compute_capped_at_one():
    """Score saturates at 1.0."""
    frag = make_fragment(content_hash="h", reuse_score=0.9)
    history = ["h"] * 10
    score = density(frag, history)
    assert score == 1.0


def test_importance_multiplier():
    """Importance multiplier scales the score."""
    frag = make_fragment(reuse_score=0.5)
    assert density(frag, [], importance=2.0) == 1.0


def test_recency_bonus():
    """Recency bonus applied when last access is this fragment."""
    frag = make_fragment(content_hash="h", reuse_score=0.4)
    history = ["x", "h"]
    score = density(frag, history)
    assert score == pytest.approx(0.4 + 1 * 0.05 + 0.1)
