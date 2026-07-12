"""Tests for AnthropicBackend."""

from unittest.mock import MagicMock

import httpx
import pytest

from membrane.compute.anthropic_backend import AnthropicBackend


class TestAnthropicBackend:
    """Test suite for Anthropic API backend."""

    @pytest.fixture
    def backend(self):
        return AnthropicBackend(api_key="sk-ant-test", model="claude-3-sonnet-20240229")

    def test_device_name(self, backend):
        assert backend.device_name() == "anthropic(claude-3-sonnet-20240229)"

    def test_prefill_returns_fragments(self, backend):
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert frags[0].content_hash is not None

    def test_generate_success(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "hello"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend._client = mock_client

        result = backend.generate([1, 2], "m")
        assert result["text"] == "hello"

    def test_generate_failure(self, backend):
        mock_client = MagicMock()
        # Simulate a network timeout.
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
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
        # Simulate a network timeout during the availability probe.
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        backend._client = mock_client
        assert backend.available() is False
