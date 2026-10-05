"""SPIFFE: TLS from the Workload API, authorization by SPIFFE ID, SVID refresh."""

import socket
import ssl
import sys
import time
import types
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from membrane.auth.spiffe import parse_id_scopes
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.transport.spiffe import SPIFFEClient, SPIFFEConfig
from tests.tls_helpers import issue, make_ca

TRUST_DOMAIN = "example.org"
NODE_ID = f"spiffe://{TRUST_DOMAIN}/membrane/node"
CLIENT_ID = f"spiffe://{TRUST_DOMAIN}/app/reader"
STRANGER_ID = f"spiffe://{TRUST_DOMAIN}/app/stranger"


class FakeWorkloadApi:
    """Stands in for the ``spiffe`` SDK: serves the current SVID from ``state``."""

    def __init__(self, ca) -> None:
        self.ca = ca
        self.rotate()

    def rotate(self) -> None:
        cert, key, self.serial = issue(self.ca, "", spiffe_id=NODE_ID)
        self.cert, self.key = cert, key

    def module(self) -> types.ModuleType:
        api = self

        class Svid:
            def __init__(self) -> None:
                self.cert_chain = [x509.load_pem_x509_certificate(api.cert.encode())]
                self.private_key = serialization.load_pem_private_key(api.key.encode(), password=None)
                self.spiffe_id = types.SimpleNamespace(trust_domain=TRUST_DOMAIN, __str__=lambda s: NODE_ID)

        class BundleSet:
            def get_bundle_for_trust_domain(self, _td: str):
                return types.SimpleNamespace(x509_authorities={api.ca[0]})

        class Client:
            def __init__(self, socket_path: str) -> None:
                assert socket_path == "/run/spire/agent.sock"

            def __enter__(self):
                return self

            def __exit__(self, *_exc) -> None:
                pass

            def fetch_x509_context(self):
                return types.SimpleNamespace(default_svid=Svid(), x509_bundle_set=BundleSet())

        return types.SimpleNamespace(WorkloadApiClient=Client)


@pytest.fixture
def workload_api(monkeypatch):
    api = FakeWorkloadApi(make_ca())
    monkeypatch.setitem(sys.modules, "spiffe", api.module())
    return api


def write_pair(directory: Path, name: str, cert: str, key: str) -> tuple[Path, Path]:
    cert_path, key_path = directory / f"{name}.crt", directory / f"{name}.key"
    cert_path.write_text(cert)
    key_path.write_text(key)
    return cert_path, key_path


def request_status(port: int, ca_pem: str, cert: Path, key: Path) -> int:
    context = ssl.create_default_context(cadata=ca_pem)
    context.check_hostname = False
    context.load_cert_chain(cert, key)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw, context.wrap_socket(raw) as tls:
        tls.sendall(b"GET /inventory HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        return int(tls.recv(64).split(b" ")[1])


def served_serial(port: int, ca_pem: str, cert: Path, key: Path) -> int:
    context = ssl.create_default_context(cadata=ca_pem)
    context.check_hostname = False
    context.load_cert_chain(cert, key)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw, context.wrap_socket(raw) as tls:
        return x509.load_der_x509_certificate(tls.getpeercert(binary_form=True)).serial_number


def test_client_builds_mtls_from_the_workload_api(workload_api) -> None:
    config = SPIFFEClient(SPIFFEConfig("/run/spire/agent.sock", frozenset({CLIENT_ID}))).fetch_mtls_config()
    assert config.require_client_cert and not config.verify_hostname
    assert config.allowed_cns == {CLIENT_ID}
    assert "BEGIN CERTIFICATE" in config.ca_bundle_pem and "BEGIN PRIVATE KEY" in config.server_key_pem


def test_missing_sdk_is_an_error_not_a_fake_config(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "spiffe", None)
    with pytest.raises(RuntimeError, match="install membrane\\[tls-spiffe\\]"):
        SPIFFEClient().fetch_mtls_config()


def test_spiffe_settings_validation() -> None:
    with pytest.raises(SettingsError, match="--tls-spiffe-allow"):
        ServerSettings(tls_spiffe_socket="/s")
    with pytest.raises(SettingsError, match="expected spiffe://"):
        ServerSettings(tls_spiffe_socket="/s", tls_spiffe_allow=("not-an-id",))
    assert parse_id_scopes([f"{CLIENT_ID}=read,write"]) == {CLIENT_ID: frozenset({"read", "write"})}


def test_serve_with_spiffe_identity(workload_api, tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server, mode = build_server(
        ServerSettings(
            host="0.0.0.0",
            port=port,
            tls_spiffe_socket="/run/spire/agent.sock",
            tls_spiffe_allow=(f"{CLIENT_ID}=read",),
            load_hooks=False,
        )
    )
    assert mode == "SPIFFE"
    ca_pem = workload_api.ca[0].public_bytes(serialization.Encoding.PEM).decode()
    reader = write_pair(tmp_path, "reader", *issue(workload_api.ca, "", spiffe_id=CLIENT_ID)[:2])
    stranger = write_pair(tmp_path, "stranger", *issue(workload_api.ca, "", spiffe_id=STRANGER_ID)[:2])
    server.start()
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                assert request_status(port, ca_pem, *reader) == 200
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.1)
        assert request_status(port, ca_pem, *stranger) == 401
        first = served_serial(port, ca_pem, *reader)
        workload_api.rotate()
        assert server.refresh_svid() is True
        assert served_serial(port, ca_pem, *reader) == workload_api.serial != first
        assert server.refresh_svid() is False  # unchanged SVID
    finally:
        server.stop(2.0)
