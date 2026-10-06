"""AWS, GCP, and Vault secret providers against stand-ins for their SDKs (optional extras)."""

import sys
import types

import pytest

from membrane.secrets import SecretBackendError, SecretNotFoundError


class ClientError(Exception):
    """botocore.exceptions.ClientError's shape."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def fake_boto3(monkeypatch, secrets: dict[str, str], error: str = "") -> list[dict]:
    sessions: list[dict] = []

    class Client:
        def get_secret_value(self, SecretId):  # noqa: N803 -- boto3's keyword
            if error:
                raise ClientError(error)
            if SecretId not in secrets:
                raise ClientError("ResourceNotFoundException")
            return {"SecretString": secrets[SecretId]}

    class Session:
        def __init__(self, profile_name=None, region_name=None):
            sessions.append({"profile_name": profile_name, "region_name": region_name})

        def client(self, service):
            assert service == "secretsmanager"
            return Client()

    boto3 = types.ModuleType("boto3")
    boto3.session = types.SimpleNamespace(Session=Session)
    botocore = types.ModuleType("botocore")
    exceptions = types.ModuleType("botocore.exceptions")
    exceptions.ClientError = ClientError
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)
    return sessions


def test_aws_reads_through_a_profile_session(monkeypatch) -> None:
    from membrane.secrets.aws import AWSSecretsProvider

    sessions = fake_boto3(monkeypatch, {"db": "hunter2"})
    provider = AWSSecretsProvider(region_name="eu-west-1", profile_name="ops")
    assert provider.get("db") == "hunter2"
    assert sessions == [{"profile_name": "ops", "region_name": "eu-west-1"}]
    with pytest.raises(SecretNotFoundError):
        provider.get("missing")
    AWSSecretsProvider().get("db")
    assert sessions[-1] == {"profile_name": None, "region_name": None}  # SDK defaults apply


def test_aws_backend_errors(monkeypatch) -> None:
    from membrane.secrets.aws import AWSSecretsProvider

    fake_boto3(monkeypatch, {}, error="AccessDeniedException")
    with pytest.raises(SecretBackendError, match="AccessDenied"):
        AWSSecretsProvider(region_name="us-east-1").get("db")


def fake_gcp(monkeypatch, secrets: dict[str, bytes]) -> list[str]:
    requested: list[str] = []

    class NotFoundError(Exception):  # google.api_core.exceptions.NotFound's role
        pass

    class Client:
        def access_secret_version(self, request):
            requested.append(request["name"])
            if request["name"].endswith("/boom/versions/latest"):
                raise RuntimeError("permission denied")
            if request["name"] not in secrets:
                raise NotFoundError(request["name"])
            return types.SimpleNamespace(payload=types.SimpleNamespace(data=secrets[request["name"]]))

    secretmanager = types.ModuleType("google.cloud.secretmanager")
    secretmanager.SecretManagerServiceClient = Client
    cloud = types.ModuleType("google.cloud")
    cloud.secretmanager = secretmanager
    google = types.ModuleType("google")
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", secretmanager)
    return requested


def test_gcp_reads_latest_versions_and_full_paths(monkeypatch) -> None:
    from membrane.secrets.gcp import GCPSecretsProvider

    path = "projects/p1/secrets/db/versions/latest"
    requested = fake_gcp(monkeypatch, {path: b"s3cret", "projects/p1/secrets/db/versions/3": b"old"})
    provider = GCPSecretsProvider(project_id="p1")
    assert provider.get("db") == "s3cret"
    assert provider.get("projects/p1/secrets/db/versions/3") == "old"
    assert requested == [path, "projects/p1/secrets/db/versions/3"]
    with pytest.raises(SecretNotFoundError):
        provider.get("absent")
    with pytest.raises(SecretBackendError, match="permission denied"):
        provider.get("boom")


def fake_hvac(monkeypatch, store: dict[str, dict]) -> list[str]:
    paths: list[str] = []

    class KV:
        def __init__(self, version: int) -> None:
            self.version = version

        def read_secret(self, path, mount_point):
            # hvac's request paths: v2 adds "/data/" itself.
            url = f"/v1/{mount_point}/data/{path}" if self.version == 2 else f"/v1/{mount_point}/{path}"
            paths.append(url)
            if path == "down":
                raise ConnectionError("vault sealed")
            data = store.get(url, {})
            return {"data": {"data": data}} if self.version == 2 else {"data": data}

    class Client:
        def __init__(self, url, token):
            self.secrets = types.SimpleNamespace(kv=types.SimpleNamespace(v1=KV(1), v2=KV(2)))

    hvac = types.ModuleType("hvac")
    hvac.Client = Client
    monkeypatch.setitem(sys.modules, "hvac", hvac)
    return paths


def test_vault_kv2_reads_the_mount_not_a_doubled_data_path(monkeypatch) -> None:
    from membrane.secrets.vault import VaultSecretProvider

    paths = fake_hvac(monkeypatch, {"/v1/secret/data/db": {"value": "pw"}, "/v1/kv/db": {"value": "v1pw"}})
    assert VaultSecretProvider(url="http://v:8200", token="t").get("db") == "pw"
    # The old default prefix still reaches the same secret.
    assert VaultSecretProvider(url="http://v:8200", token="t", path_prefix="secret/data").get("db") == "pw"
    assert paths == ["/v1/secret/data/db", "/v1/secret/data/db"]
    assert VaultSecretProvider(url="http://v:8200", token="t", path_prefix="kv", kv_version=1).get("db") == "v1pw"
    with pytest.raises(SecretNotFoundError):
        VaultSecretProvider(url="http://v:8200", token="t").get("nothing")
    with pytest.raises(SecretBackendError, match="sealed"):
        VaultSecretProvider(url="http://v:8200", token="t").get("down")
