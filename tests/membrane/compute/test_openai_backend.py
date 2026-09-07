"""Tests for OpenAI."""

from unittest.mock import MagicMock

import httpx
import pytest

from membrane.compute.openai import OpenAI
from membrane.fragment import Fragment


class TestOpenAIBackend:
    """Test suite for OpenAI API backend."""

    @pytest.fixture
    def backend(self):
        return OpenAI(api_key="sk-test", model="gpt-4o-mini")

    def test_device_name(self, backend):
        assert backend.device_name() == "openai(gpt-4o-mini)"

    def test_prefill_uses_mock_embedding(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend.client = mock_client

        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert isinstance(frags[0], Fragment)
        # The new Fragment schema drops ``embedding``; verify the
        # identity's payload_hash is set instead.
        assert frags[0].identity.payload_hash
        mock_client.post.assert_called_once()

    def test_prefill_fallback_when_client_none(self, backend):
        backend.client = None
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert frags[0].identity.payload_hash

    def test_generate_success(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend.client = mock_client

        result = backend.generate([1, 2], "m")
        assert result["text"] == "hi"

    def test_generate_failure(self, backend):
        mock_client = MagicMock()
        # Simulate an HTTP error returned by the server (e.g. 429).
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "rate limit",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        backend.client = mock_client

        result = backend.generate([1, 2], "m")
        assert result["text"] == ""

    def test_available_when_healthy(self, backend):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        backend.client = mock_client
        assert backend.available() is True

    def test_available_when_unhealthy(self, backend):
        mock_client = MagicMock()
        # Simulate a network timeout during the availability probe.
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        backend.client = mock_client
        assert backend.available() is False
