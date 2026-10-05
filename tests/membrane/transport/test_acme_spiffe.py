"""Tests for the SPIFFE adapter."""

import pytest

from membrane.transport.spiffe import SPIFFEClient, SPIFFEConfig


class TestSPIFFEClient:
    def test_defaults(self):
        config = SPIFFEConfig()
        assert config.socket_path == "/run/spiffe/workload-api.sock"
