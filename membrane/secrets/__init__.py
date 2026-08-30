"""SecretProvider Protocol + EnvSecretProvider (Phase 3.4.5).

The v2.0 release carried API keys + mTLS PEMs as plain
:class:`MTLSConfig` fields and read them straight from
``sys.argv`` / environment variables. The v3.0.0 release
introduces a pluggable :class:`SecretProvider` Protocol with
the following implementations (each gated on a separate
optional dependency):

* :class:`EnvSecretProvider` (default; no deps): reads from
  environment variables via the standard library.
* :class:`VaultSecretProvider` (Phase 3.4.5b; ``pip install
  membrane[secrets-vault]``): reads from HashiCorp Vault via
  ``hvac``.
* :class:`AWSSecretsProvider`` (Phase 3.4.5c; ``pip install
  membrane[secrets-aws]``): reads from AWS Secrets Manager
  via ``boto3``.
* :class:`GCPSecretsProvider`` (Phase 3.4.5d; ``pip install
  membrane[secrets-gcp]``): reads from Google Secret Manager
  via ``google-cloud-secret-manager``.

The :data:`get_default_provider` accessor returns the
process-wide provider; operators install their backend via
:func:`set_default_provider` at startup.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SecretProvider(Protocol):
    """Pluggable secret backend.

    Implementations must:

    * Return the requested secret as ``str`` on success.
    * Raise :class:`SecretNotFoundError` when the secret
      identifier is unknown.
    * Raise :class:`SecretBackendError` for any other backend
      failure (network, IAM, etc.).
    """

    def get(self, secret_name: str) -> str:
        """Look up ``secret_name``.

        Args:
            secret_name: Backend-specific identifier (env var
                name, Vault path, AWS arn, GCP resource).

        Returns:
            str: The secret payload.
        """
        ...


class SecretNotFoundError(KeyError):
    """Raised when the requested secret identifier is unknown."""


class SecretBackendError(RuntimeError):
    """Raised for non-recoverable secret-backend failures."""


@dataclass(frozen=True)
class EnvSecretProvider:
    """Reads from process environment variables.

    Attributes:
        env: Mapping to read from; defaults to ``os.environ``.
    """

    env: dict[str, str] | None = None

    def get(self, secret_name: str) -> str:
        """Look up ``secret_name`` in the configured environment.

        Args:
            secret_name: The environment variable name.

        Returns:
            str: The env var value.

        Raises:
            SecretNotFoundError: When the env var is unset.
        """
        mapping = self.env if self.env is not None else os.environ
        try:
            return mapping[secret_name]
        except KeyError as exc:
            raise SecretNotFoundError(secret_name) from exc


class _LazyProxy:
    """Proxy that constructs the underlying provider on first call.

    The wrapper exists so :func:`set_default_provider` can
    store a factory that defers its own dependency import; the
    real provider is constructed lazily.
    """

    def __init__(self, factory: type[SecretProvider]) -> None:
        self._factory = factory
        self._instance: SecretProvider | None = None

    def get(self, secret_name: str) -> str:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance.get(secret_name)


_DEFAULT_PROVIDER: SecretProvider | None = None


def get_default_provider() -> SecretProvider:
    """Return the process-wide default :class:`SecretProvider`.

    Returns:
        SecretProvider: The provider installed via
        :func:`set_default_provider`, or a fresh
        :class:`EnvSecretProvider` when nothing is installed
        (single-node / test deployments).
    """
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = EnvSecretProvider()
    return _DEFAULT_PROVIDER


def set_default_provider(provider: SecretProvider) -> None:
    """Replace the process-wide default :class:`SecretProvider`.

    Args:
        provider: The new provider.
    """
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider


def reset_default_provider() -> None:
    """Restore the process-wide provider to its factory default.

    Tests call this to undo a :func:`set_default_provider`
    without leaking policy into other test cases.
    """
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = None


__all__ = [
    "EnvSecretProvider",
    "SecretBackendError",
    "SecretNotFoundError",
    "SecretProvider",
    "get_default_provider",
    "reset_default_provider",
    "set_default_provider",
]
