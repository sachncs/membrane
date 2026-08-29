from tests.conftest import make_fragment

"""Tests for predictor module."""

import pytest

from membrane.node import Node
from membrane.predict import Predict


class TestPredictor:
    """Test suite for Predict."""

    def test_predict_kv_size_positive(self):
        p = Predict()
        assert p.predict_kv_size(list(range(100))) > 0.0

    def test_predict_kv_size_bias(self):
        p1 = Predict(kv_size_bias=1.0)
        p2 = Predict(kv_size_bias=2.0)
        tokens = list(range(100))
        assert p2.predict_kv_size(tokens) == pytest.approx(p1.predict_kv_size(tokens) * 2.0)

    def test_predict_reuse_empty_history(self):
        p = Predict()
        assert p.predict_reuse_probability("h", []) == 0.0

    def test_predict_reuse_with_history(self):
        p = Predict()
        history = ["a", "b", "h", "h"]
        prob = p.predict_reuse_probability("h", history)
        assert prob == 0.5

    def test_predict_reuse_capped_at_one(self):
        p = Predict()
        history = ["h"] * 10
        assert p.predict_reuse_probability("h", history) == 1.0

    def test_predict_optimal_region_empty(self):
        p = Predict()
        assert p.predict_optimal_region(list(range(10)), []) == ""

    def test_predict_optimal_region_lowest_load(self):
        p = Predict()
        n1 = Node("n1", max_memory_bytes=100)
        n2 = Node("n2", max_memory_bytes=100)

        f = make_fragment("x", size=90)
        n1.store(f, is_primary=True)
        best = p.predict_optimal_region(list(range(10)), [n1, n2])
        assert best == "n2"
