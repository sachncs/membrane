"""Tests for the SecretProvider abstraction (Phase 3.4.5)."""

from __future__ import annotations

import pytest

from membrane.secrets import (
    EnvSecretProvider,
    SecretBackendError,
    SecretNotFoundError,
    SecretProvider,
    get_default_provider,
    reset_default_provider,
    set_default_provider,
)


class TestEnvSecretProvider:
    def test_get_returns_value(self):
        provider = EnvSecretProvider(env={"FOO": "bar"})
        assert provider.get("FOO") == "bar"

    def test_missing_secret_raises(self):
        provider = EnvSecretProvider(env={})
        with pytest.raises(SecretNotFoundError):
            provider.get("MISSING")

    def test_default_uses_os_environ(self):
        import os

        os.environ["_TEST_SECRET_FOR_DEFAULT"] = "x"
        try:
            provider = EnvSecretProvider()
            assert provider.get("_TEST_SECRET_FOR_DEFAULT") == "x"
        finally:
            del os.environ["_TEST_SECRET_FOR_DEFAULT"]


class TestProcessWideDefault:
    def setup_method(self):
        reset_default_provider()

    def teardown_method(self):
        reset_default_provider()

    def test_default_falls_back_to_env(self):
        provider = get_default_provider()
        assert isinstance(provider, EnvSecretProvider)

    def test_set_and_reset(self):
        class Fake:
            def get(self, secret_name: str) -> str:
                return f"fake:{secret_name}"

        fake = Fake()
        set_default_provider(fake)  # type: ignore[arg-type]
        assert get_default_provider() is fake
        reset_default_provider()
        assert not isinstance(get_default_provider(), Fake)


class TestProtocolCompliance:
    def test_env_provider_satisfies_protocol(self):
        provider = EnvSecretProvider(env={"A": "1"})
        assert isinstance(provider, SecretProvider)


class TestOptionalBackends:
    def test_vault_provider_requires_hvac(self):
        from membrane.secrets.vault import VaultSecretProvider

        with pytest.raises(SecretBackendError):
            VaultSecretProvider(url="http://x", token="t")

    def test_aws_provider_requires_boto3(self):
        from membrane.secrets.aws import AWSSecretsProvider

        with pytest.raises(SecretBackendError):
            AWSSecretsProvider(region_name="us-east-1")

    def test_gcp_provider_requires_google_cloud(self):
        from membrane.secrets.gcp import GCPSecretsProvider

        with pytest.raises(SecretBackendError):
            GCPSecretsProvider(project_id="proj")
