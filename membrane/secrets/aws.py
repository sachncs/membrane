"""AWS Secrets Manager provider.

Pulls secrets from AWS Secrets Manager via :mod:`boto3`.
The provider is installed via ``pip install
membrane[secrets-aws]``; the import is lazy so the absence
of :mod:`boto3` raises a clear error only when
:class:`AWSSecretsProvider` is actually instantiated.

Attributes:
    region_name: AWS region for the Secrets Manager endpoint.
    profile_name: Optional boto3 profile.
"""

import logging
from dataclasses import dataclass
from typing import override

from membrane.secrets import SecretBackendError, SecretNotFoundError, SecretProvider

logger = logging.getLogger(__name__)


@dataclass
class AWSSecretsProvider(SecretProvider):
    """AWS Secrets Manager-backed secret provider.

    Attributes:
        region_name: AWS region for the Secrets Manager endpoint.
        profile_name: Optional boto3 profile name.
    """

    region_name: str = ""
    profile_name: str | None = None

    def __post_init__(self) -> None:
        """Verify :mod:`boto3` is importable when constructed.

        Raises:
            SecretBackendError: When boto3 is not installed.
        """
        try:
            import boto3  # noqa: F401  -- presence probe.
        except ImportError as exc:
            raise SecretBackendError("AWSSecretsProvider requires 'boto3'; install membrane[secrets-aws]") from exc

    @override
    def get(self, secret_name: str) -> str:
        """Read ``secret_name`` from AWS Secrets Manager.

        Args:
            secret_name: The full secret ARN or name.

        Returns:
            str: The plaintext secret value.
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as exc:  # pragma: no cover - guarded above
            raise SecretBackendError("boto3 not installed") from exc
        # A profile is a Session setting; boto3.client() does not accept one.
        session = boto3.session.Session(profile_name=self.profile_name or None, region_name=self.region_name or None)
        client = session.client("secretsmanager")
        try:
            response = client.get_secret_value(SecretId=secret_name)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                raise SecretNotFoundError(secret_name) from exc
            raise SecretBackendError(f"aws secrets error for {secret_name}: {exc}") from exc
        return response.get("SecretString") or ""


__all__ = ["AWSSecretsProvider"]
