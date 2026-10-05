"""SecretProvider Protocol + EnvSecretProvider.

The v2.0 release carried API keys + mTLS PEMs as plain
:class:`MTLSConfig` fields and read them straight from
``sys.argv`` / environment variables. The v3.0.0 release
introduces a pluggable :class:`SecretProvider` Protocol with
the following implementations (each gated on a separate
optional dependency):

* :class:`EnvSecretProvider` (default; no deps): reads from
  environment variables via the standard library.
* :class:`VaultSecretProvider` (``pip install
  membrane[secrets-vault]``): reads from HashiCorp Vault via
  ``hvac``.
* :class:`AWSSecretsProvider`` (``pip install
  membrane[secrets-aws]``): reads from AWS Secrets Manager
  via ``boto3``.
* :class:`GCPSecretsProvider`` (``pip install
  membrane[secrets-gcp]``): reads from Google Secret Manager
  via ``google-cloud-secret-manager``.

The :data:`get_default_provider` accessor returns the
process-wide provider; operators install their backend via
:func:`set_default_provider` at startup.
"""

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


class LazyProxy:
    """Proxy that constructs the underlying provider on first call.

    The wrapper exists so :func:`set_default_provider` can
    store a factory that defers its own dependency import; the
    real provider is constructed lazily.
    """

    def __init__(self, factory: type[SecretProvider]) -> None:
        """Wrap a provider class that is instantiated on first use.

        Args:
            factory: Provider class instantiated on first use.
        """
        self.__factory = factory
        self.__instance: SecretProvider | None = None

    def get(self, secret_name: str) -> str:
        """Return a secret, creating the provider on first use.

        Args:
            secret_name: Backend-specific identifier (env var name, Vault path,
                AWS arn, GCP resource).

        Returns:
            str: A secret, creating the provider on first use.
        """
        if self.__instance is None:
            self.__instance = self.__factory()
        return self.__instance.get(secret_name)


#: Prefix that marks a setting as a reference into the secret provider.
SECRET_SCHEME = "secret://"

DEFAULT_PROVIDER: SecretProvider | None = None


def get_default_provider() -> SecretProvider:
    """Return the process-wide default :class:`SecretProvider`.

    Returns:
        SecretProvider: The provider installed via
        :func:`set_default_provider`, or a fresh
        :class:`EnvSecretProvider` when nothing is installed
        (single-node / test deployments).
    """
    global DEFAULT_PROVIDER
    if DEFAULT_PROVIDER is None:
        DEFAULT_PROVIDER = EnvSecretProvider()
    return DEFAULT_PROVIDER


def set_default_provider(provider: SecretProvider) -> None:
    """Replace the process-wide default :class:`SecretProvider`.

    Args:
        provider: The new provider.
    """
    global DEFAULT_PROVIDER
    DEFAULT_PROVIDER = provider


def is_secret_ref(value: str) -> bool:
    """Whether ``value`` is a ``secret://name`` reference.

    Args:
        value: A setting value.

    Returns:
        bool: True for ``secret://`` references.
    """
    return value.startswith(SECRET_SCHEME)


def resolve_secret(ref: str, provider: SecretProvider | None = None) -> str:
    """Fetch the secret a ``secret://name`` reference points at.

    Args:
        ref: The reference.
        provider: Provider to ask; the process default when ``None``.

    Returns:
        str: The secret.

    Raises:
        ValueError: When ``ref`` is not a ``secret://`` reference or names
            nothing.
        SecretNotFoundError: When the provider has no such secret.
        SecretBackendError: When the provider fails.
    """
    if not is_secret_ref(ref) or len(ref) == len(SECRET_SCHEME):
        raise ValueError(f"not a secret reference: {ref!r}")
    return (provider or get_default_provider()).get(ref[len(SECRET_SCHEME) :])


def reset_default_provider() -> None:
    """Restore the process-wide provider to its factory default.

    Tests call this to undo a :func:`set_default_provider`
    without leaking policy into other test cases.
    """
    global DEFAULT_PROVIDER
    DEFAULT_PROVIDER = None


__all__ = [
    "SECRET_SCHEME",
    "EnvSecretProvider",
    "SecretBackendError",
    "SecretNotFoundError",
    "SecretProvider",
    "get_default_provider",
    "is_secret_ref",
    "reset_default_provider",
    "resolve_secret",
    "set_default_provider",
]
