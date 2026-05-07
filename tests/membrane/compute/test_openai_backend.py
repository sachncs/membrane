"""Tests for OpenAIBackend."""

from unittest.mock import MagicMock

import pytest

from membrane.compute.openai_backend import OpenAIBackend
from membrane.fragment import Fragment


class TestOpenAIBackend:
    """Test suite for OpenAI API backend."""

    @pytest.fixture
    def backend(self):
        return OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")

    def test_device_name(self, backend):
        assert backend.device_name() == "openai(gpt-4o-mini)"

    def test_prefill_uses_mock_embedding(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend._client = mock_client

        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert isinstance(frags[0], Fragment)
        assert len(frags[0].embedding) == 4
        mock_client.post.assert_called_once()

    def test_prefill_fallback_when_client_none(self, backend):
        backend._client = None
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert frags[0].embedding == (0.0, 4.0)

    def test_generate_success(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend._client = mock_client

        result = backend.generate([1, 2], "m")
        assert result["text"] == "hi"

    def test_generate_failure(self, backend):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("rate limit")
        backend._client = mock_client

        result = backend.generate([1, 2], "m")
        assert result["text"] == ""

    def test_available_when_healthy(self, backend):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        backend._client = mock_client
        assert backend.available() is True

    def test_available_when_unhealthy(self, backend):
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("timeout")
        backend._client = mock_client
        assert backend.available() is False
