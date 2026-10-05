"""ServerSettings validation and the startup policy in build_server."""

import pytest
from typer.testing import CliRunner

from membrane.auth.apikey import generate_key
from membrane.cli import app
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.transport.limits import TransportLimits


def private_file(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    path.chmod(0o600)
    return str(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"port": 70000}, "out of range"),
        ({"transport": "grpc"}, "only 'http'"),
        ({"compute": "nope"}, "unknown compute backend"),
        ({"consistency": "maybe"}, "consistency must be one of"),
        ({"peer_networks": ("not-a-cidr",)}, "not a CIDR"),
        ({"drain_timeout": -1.0}, "drain timeout"),
        ({"limits": TransportLimits(rate_limit_per_sec=-1)}, "rate limit"),
        ({"api_key_file": "a", "auth_config": "b"}, "not both"),
    ],
)
def test_invalid_settings_are_rejected(overrides, message) -> None:
    with pytest.raises(SettingsError, match=message):
        ServerSettings(**overrides)


def test_settings_hide_llm_key_from_repr() -> None:
    assert "secret-llm-key" not in repr(ServerSettings(llm_api_key="secret-llm-key"))


def test_public_bind_requires_auth() -> None:
    with pytest.raises(SettingsError, match="Refusing to serve unauthenticated"):
        build_server(ServerSettings(host="0.0.0.0", port=0))
    _, mode = build_server(ServerSettings(host="0.0.0.0", port=0, allow_unauthenticated=True))
    assert mode.startswith("NONE")


def test_hashed_keyfile_builds_authenticated_server(tmp_path) -> None:
    _, line = generate_key("svc", ["read"])
    server, mode = build_server(
        ServerSettings(host="0.0.0.0", port=0, api_key_file=private_file(tmp_path, "keys", line + "\n"))
    )
    assert mode == "API key"
    assert server.authenticator is not None


def test_cluster_peer_key_needs_admin(tmp_path) -> None:
    peer_key, peer_line = generate_key("peer", ["read", "write"])
    keyfile = private_file(tmp_path, "keys", peer_line + "\n")
    peer_file = private_file(tmp_path, "peer", peer_key)
    settings = ServerSettings(port=0, peers=("127.0.0.1:9",), api_key_file=keyfile, peer_api_key_file=peer_file)
    with pytest.raises(SettingsError, match="admin"):
        build_server(settings)
    with pytest.raises(SettingsError, match="--peer-api-key-file"):
        build_server(ServerSettings(port=0, peers=("127.0.0.1:9",), api_key_file=keyfile))


def test_world_readable_data_key_is_refused(tmp_path) -> None:
    key = tmp_path / "data.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o644)
    with pytest.raises(SettingsError, match="chmod 600"):
        build_server(ServerSettings(port=0, data_dir=str(tmp_path / "d"), data_key_file=str(key)))


def test_unknown_content_store_is_reported(tmp_path) -> None:
    with pytest.raises(SettingsError, match="unknown content store"):
        build_server(ServerSettings(port=0, data_dir=str(tmp_path), content_store="s3"))


def test_keys_generate_cli_outputs_key_and_line() -> None:
    result = CliRunner().invoke(app, ["keys", "generate", "--subject", "svc", "--scope", "write"])
    assert result.exit_code == 0
    key, line = result.stdout.strip().splitlines()
    assert key.startswith("mbr_")
    assert line.startswith("sha256:") and line.endswith(":svc:write")
    assert key not in line
