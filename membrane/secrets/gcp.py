"""GCP Secret Manager provider (Phase 3.4.5d).

Pulls secrets from Google Secret Manager via
``google-cloud-secret-manager``. The provider is installed
via ``pip install membrane[secrets-gcp]``; the import is
lazy so the absence of the dependency raises a clear error
only when :class:`GCPSecretsProvider` is actually
instantiated.

Attributes:
    project_id: GCP project id (e.g., ``"membrane-prod"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from membrane.secrets import SecretBackendError, SecretNotFoundError, SecretProvider

logger = logging.getLogger(__name__)


@dataclass
class GCPSecretsProvider(SecretProvider):
    """Google Secret Manager-backed secret provider.

    Attributes:
        project_id: GCP project id.
    """

    project_id: str = ""

    def __post_init__(self) -> None:
        """Verify the GCP SDK is importable when constructed.

        Raises:
            SecretBackendError: When google-cloud-secret-manager
                is not installed.
        """
        try:
            from google.cloud import secretmanager  # noqa: F401  -- presence probe.
        except ImportError as exc:
            raise SecretBackendError(
                "GCPSecretsProvider requires 'google-cloud-secret-manager'; install membrane[secrets-gcp]"
            ) from exc

    def get(self, secret_name: str) -> str:
        """Read ``secret_name`` from Google Secret Manager.

        Args:
            secret_name: Bare name or full resource path.

        Returns:
            str: The latest payload as UTF-8 text.
        """
        try:
            from google.cloud import secretmanager
        except ImportError as exc:  # pragma: no cover - guarded above
            raise SecretBackendError("gcp SDK not installed") from exc
        try:
            client = secretmanager.SecretManagerServiceClient()
            if "/" in secret_name:
                resource = secret_name
            else:
                resource = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": resource})
        except Exception as exc:  # broad: any client error
            if "NotFound" in type(exc).__name__ or "not_found" in str(exc).lower():
                raise SecretNotFoundError(secret_name) from exc
            raise SecretBackendError(f"gcp secrets error for {secret_name}: {exc}") from exc
        return response.payload.data.decode("utf-8")


__all__ = ["GCPSecretsProvider"]
