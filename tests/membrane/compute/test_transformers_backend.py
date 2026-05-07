"""Tests for TransformersBackend."""

from unittest.mock import MagicMock, patch

import pytest

from membrane.compute.transformers_backend import TransformersBackend
from membrane.fragment import Fragment


class TestTransformersBackend:
    """Test suite for HuggingFace Transformers backend."""

    def test_device_name_unloaded(self):
        backend = TransformersBackend(model_id="gpt2")
        backend._model = None
        assert backend.device_name() == "transformers(unloaded)"

    def test_prefill_simulation_when_model_none(self):
        backend = TransformersBackend(model_id="gpt2")
        backend._model = None
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert frags[0].embedding == (0.0, 4.0)

    def test_generate_when_model_none(self):
        backend = TransformersBackend(model_id="gpt2")
        backend._model = None
        result = backend.generate([1, 2], "m")
        assert result["text"] == ""

    def test_generate_success(self):
        backend = TransformersBackend(model_id="gpt2")

        mock_tensor = MagicMock()
        mock_tensor.tolist.return_value = [100, 101, 102]
        mock_tensor.__getitem__ = MagicMock(return_value=mock_tensor)

        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=mock_tensor)

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "hello world"
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 2])}

        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._actual_device = "cpu"

        result = backend.generate([1, 2], "m")
        assert result["text"] == "hello world"
        assert result["tokens"] == [100, 101, 102]

    def test_available_when_loaded(self):
        backend = TransformersBackend(model_id="gpt2")
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        assert backend.available() is True

    def test_available_when_unloaded(self):
        backend = TransformersBackend(model_id="gpt2")
        backend._model = None
        assert backend.available() is False
