"""Tests for Ollama."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from membrane.compute.ollama import Ollama
from membrane.fragment import Fragment


class TestOllamaBackend:
    """Test suite for Ollama API backend."""

    @pytest.fixture
    def backend(self):
        return Ollama(base_url="http://localhost:11434", model="llama3.2")

    def test_device_name(self, backend):
        assert backend.device_name() == "ollama(llama3.2)"

    def test_prefill_uses_mock_embedding(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend.client = mock_client

        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert isinstance(frags[0], Fragment)
        assert frags[0].identity.payload_hash
        mock_client.post.assert_called_once()

    def test_prefill_fallback_when_client_none(self, backend):
        backend.client = None
        frags = backend.prefill([1, 2, 3, 4], "m")
        assert len(frags) == 1
        assert frags[0].identity.payload_hash

    def test_generate_success(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "hello world"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        backend.client = mock_client

        result = backend.generate([1, 2], "m")
        assert result["text"] == "hello world"

    def test_generate_failure(self, backend):
        mock_client = MagicMock()
        # Simulate a network timeout. httpx.TimeoutException is the
        # concrete exception httpx raises for read/write timeouts.
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
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
        # Simulate a refused connection with httpx.ConnectError.
        mock_client.get.side_effect = httpx.ConnectError("connection refused")
        backend.client = mock_client
        assert backend.available() is False
