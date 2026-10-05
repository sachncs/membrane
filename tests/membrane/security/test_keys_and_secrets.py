"""secret:// references, versioned data keys, rotation, and re-encryption."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from membrane.cli import app
from membrane.content_store import FilesystemBlob
from membrane.node import Node
from membrane.runtime.components import build_content_store, load_data_key
from membrane.runtime.settings import ServerSettings, SettingsError, build_server, read_secret
from membrane.secrets import EnvSecretProvider, reset_default_provider, resolve_secret, set_default_provider
from membrane.security.keyring import DirectoryKeyring, write_next_key
from membrane.server import Server


@pytest.fixture(autouse=True)
def _provider():
    yield
    reset_default_provider()


def test_secret_references_resolve_through_the_provider() -> None:
    set_default_provider(EnvSecretProvider(env={"API_KEYS": "k:svc:read\n"}))
    assert resolve_secret("secret://API_KEYS") == "k:svc:read\n"
    assert read_secret("secret://API_KEYS", "API keyfile") == "k:svc:read\n"
    with pytest.raises(SettingsError, match="cannot resolve"):
        read_secret("secret://MISSING", "API keyfile")
    with pytest.raises(ValueError):
        resolve_secret("secret://")


def test_server_settings_resolve_secret_keyfile(monkeypatch) -> None:
    monkeypatch.setenv("MEMBRANE_TEST_KEYS", "k:svc:read\n")
    _, mode = build_server(
        ServerSettings(host="0.0.0.0", port=0, api_key_file="secret://MEMBRANE_TEST_KEYS", load_hooks=False)
    )
    assert mode == "API key"
    with pytest.raises(SettingsError, match="unknown secret provider"):
        ServerSettings(secret_provider="keychain")


def test_data_key_from_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_KEY", "ab" * 32)
    store = build_content_store(str(tmp_path), "secret://DATA_KEY")
    store.put("k" * 8, b"payload")
    assert store.get("k" * 8) == b"payload"


def test_key_directory_rotation_and_reencryption(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    write_next_key(keys)
    store = build_content_store(str(tmp_path / "data"), str(keys))
    assert isinstance(store.key_provider, DirectoryKeyring)
    store.put("blob-one", b"first")

    newest = write_next_key(keys)
    assert newest.name == "v2.key" and (os.stat(newest).st_mode & 0o777) == 0o600
    assert store.key_provider.refresh() == [2]
    assert store.get("blob-one") == b"first"  # old version still decrypts
    assert store.reencrypt_all() == 1
    assert store.reencrypt_all() == 0  # idempotent

    (keys / "v1.key").unlink()  # retire the old key
    reopened = FilesystemBlob(tmp_path / "data" / "blobs", tenant_id="membrane", key_provider=DirectoryKeyring(keys))
    assert reopened.get("blob-one") == b"first"


def test_server_refreshes_key_directory(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    write_next_key(keys)
    node = Node("k0", content_store=build_content_store(str(tmp_path / "data"), str(keys)))
    node.content_store.put("blob-two", b"second")
    server = Server(node=node, port=0, load_hooks=False)
    assert server.refresh_data_keys() == 0
    write_next_key(keys)
    assert server.refresh_data_keys() == 1


def test_put_from_file_encrypts(tmp_path: Path) -> None:
    store = FilesystemBlob(tmp_path / "blobs", tenant_id="t", key_provider=load_data_key(tmp_path, ""))
    source = tmp_path / "src.bin"
    source.write_bytes(b"plain bytes")
    store.put_from_file("file-key", str(source))
    assert store.get("file-key") == b"plain bytes"
    assert b"plain bytes" not in next((tmp_path / "blobs").rglob("*.blob")).read_bytes()


def test_rotate_data_key_cli(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["keys", "rotate-data-key", str(tmp_path / "k")])
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("v1.key")
