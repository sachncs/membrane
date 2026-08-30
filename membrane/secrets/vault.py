"""VaultSecretProvider (Phase 3.4.5b).

Pulls secrets from HashiCorp Vault via the :mod:`hvac`
client. The provider is installed via ``pip install
membrane[secrets-vault]``; the import is lazy so the
absence of :mod:`hvac` raises a clear error only when a
:calss:`VaultSecretProvider` is actually instantiated.

Attributes:
    url: Vault server URL.
    token: Vault token. Operators that prefer the AWS / GCP /
    Kubernetes auth backends should swap this for the
    matching hvac client auth path; v3.0 ships the token
    backend only.
    path_prefix: KV v2 path prefix; default ``"secret/data"``.
    kv_version: KV engine version (``1`` or ``2``); default ``2``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from membrane.secrets import SecretBackendError, SecretNotFoundError, SecretProvider

logger = logging.getLogger(__name__)


@dataclass
class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault-backed secret provider.

    Attributes:
        url: Vault server URL.
        token: Vault token.
        path_prefix: KV v2 path prefix; default ``"secret/data"``.
        kv_version: KV engine version (``1`` or ``2``); default ``2``.
    """

    url: str = ""
    token: str = ""
    path_prefix: str = "secret/data"
    kv_version: int = 2

    def __post_init__(self) -> None:
        """Verify :mod:`hvac` is importable when constructed.

        Raises:
            SecretBackendError: When hvac is not installed.
        """
        try:
            import hvac  # noqa: F401  -- presence probe.
        except ImportError as exc:
            raise SecretBackendError(
                "VaultSecretProvider requires the 'hvac' package; install membrane[secrets-vault]"
            ) from exc

    def get(self, secret_name: str) -> str:
        """Read ``secret_name`` from Vault.

        Args:
            secret_name: KV path (``prefix`` + name).

        Returns:
            str: The ``value`` field of the latest version.
        """
        try:
            import hvac
        except ImportError as exc:  # pragma: no cover - guarded above
            raise SecretBackendError("hvac not installed") from exc
        client: Any = hvac.Client(url=self.url, token=self.token)
        try:
            if self.kv_version == 2:
                response = client.secrets.kv.v2.read_secret(
                    path=secret_name, mount_point=self.path_prefix
                )
                data = response.get("data", {}).get("data", {})
            else:
                response = client.secrets.kv.v1.read_secret(
                    path=secret_name, mount_point=self.path_prefix
                )
                data = response.get("data", {})
        except Exception as exc:
            raise SecretBackendError(f"vault read failed for {secret_name}: {exc}") from exc
        if not data or "value" not in data:
            raise SecretNotFoundError(secret_name)
        return str(data["value"])


__all__ = ["VaultSecretProvider"]
