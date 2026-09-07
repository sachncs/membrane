"""Tests for the ACME and SPIFFE adapters (Phase 3.4.3 + 3.4.4)."""

from __future__ import annotations

import pytest

from membrane.transport.acme import ACMEClient, ACMEConfig
from membrane.transport.spiffe import SPIFFEClient, SPIFFEConfig


class TestACMEConfig:
    def test_defaults(self):
        config = ACMEConfig()
        assert config.directory_url == ""
        assert config.contact is None


class TestACMEClient:
    def test_construct_with_dependencies(self):
        """cryptography is installed in the v3.0.0 image; the
        client constructs without raising."""
        client = ACMEClient(ACMEConfig(directory_url="http://acme/dir"))
        assert client.config.directory_url == "http://acme/dir"

    def test_issue_not_implemented(self):
        """The v3.0.0 release ships the class surface; the
        HTTP-01 + DNS-01 challenge handlers are 3.0.1."""
        client = ACMEClient(ACMEConfig(directory_url="http://acme/dir"))
        with pytest.raises(NotImplementedError):
            client.issue_certificate(["example.com"], "/tmp/cert.pem")

    def test_poll_not_implemented(self):
        client = ACMEClient(ACMEConfig())
        with pytest.raises(NotImplementedError):
            client.poll_until_ready(type(client).__new__(client.__class__))


class TestSPIFFEClient:
    def test_defaults(self):
        config = SPIFFEConfig()
        assert config.socket_path == "/run/spiffe/workload-api.sock"

    def test_fallback_when_spiffe_sdk_missing(self):
        """Without the SPIFFE SDK the client returns an empty MTLSConfig fallback."""
        from membrane.transport.tls import MTLSConfig

        client = SPIFFEClient(SPIFFEConfig())
        config = client.fetch_mtls_config()
        # The fallback returns placeholder PEMs; the import
        # surfaces a fresh MTLSConfig so the call never raises.
        assert isinstance(config, MTLSConfig)
