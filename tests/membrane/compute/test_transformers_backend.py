"""Tests for Transformers."""

from unittest.mock import MagicMock, patch

import pytest

from membrane.compute.transformers import Transformers
from membrane.fragment import Fragment

# Skip the success-path test when torch is not installed. The
# generate() implementation imports torch to set up the
# torch.no_grad() context manager; without torch the call falls
# back to the except branch and returns empty results, which
# would mask real regressions in the success path.
torch = pytest.importorskip("torch")


class TestTransformersBackend:
    """Test suite for HuggingFace Transformers backend."""

    def test_device_name_unloaded(self):
        backend = Transformers(model_id="gpt2")
        backend.model = None
        assert backend.device_name() == "transformers(unloaded)"

    def test_prefill_simulation_when_model_none(self):
        backend = Transformers(model_id="gpt2")
        backend.model = None
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        # Surrogate fallback stamps the token span onto the
        # identity; the synthetic embedding surrogate is gone
        # in the v2 schema.
        assert frags[0].identity.token_span == (0, 3)
        assert frags[0].identity.model_id == "m"

    def test_generate_when_model_none(self):
        backend = Transformers(model_id="gpt2")
        backend.model = None
        result = backend.generate([1, 2], "m")
        assert result["text"] == ""

    def test_generate_success(self):
        backend = Transformers(model_id="gpt2")

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

        backend.model = mock_model
        backend.tokenizer = mock_tokenizer

        result = backend.generate([1, 2], "m")
        assert result["text"] == "hello world"
        assert result["tokens"] == [100, 101, 102]

    def test_available_when_loaded(self):
        backend = Transformers(model_id="gpt2")
        backend.model = MagicMock()
        backend.tokenizer = MagicMock()
        assert backend.available() is True

    def test_available_when_unloaded(self):
        backend = Transformers(model_id="gpt2")
        backend.model = None
        assert backend.available() is False
