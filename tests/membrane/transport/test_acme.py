"""ACME client: JWS signing, HTTP-01 responder, renewal window, and Pebble end to end."""

import base64
import json
import os
import socket
import ssl
import time
import urllib.request
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.transport.acme import ACMEClient, ACMEConfig, ChallengeResponder, ensure_certificate
from tests.tls_helpers import issue, make_ca


def unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def test_jws_is_es256_signed_by_the_account_key(tmp_path: Path) -> None:
    client = ACMEClient(ACMEConfig(state_dir=str(tmp_path)))
    client._ACMEClient__nonce = "nonce-1"  # skip the network round trip
    body = json.loads(client.sign("https://acme.test/new-order", {"a": 1}))
    protected = json.loads(unb64(body["protected"]))
    assert protected == {"alg": "ES256", "nonce": "nonce-1", "url": "https://acme.test/new-order", "jwk": client.jwk()}
    raw = unb64(body["signature"])
    signature = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    client.account_key().public_key().verify(
        signature, f"{body['protected']}.{body['payload']}".encode(), ec.ECDSA(hashes.SHA256())
    )
    assert (os.stat(tmp_path / "account.key").st_mode & 0o777) == 0o600
    # The account key persists, so the thumbprint is stable across clients.
    assert ACMEClient(ACMEConfig(state_dir=str(tmp_path))).thumbprint() == client.thumbprint()
    client.close()


def test_responder_serves_only_open_tokens() -> None:
    with ChallengeResponder(0, host="127.0.0.1") as responder:
        responder.tokens["tok"] = "tok.thumb"
        base = f"http://127.0.0.1:{responder.port}/.well-known/acme-challenge/"
        assert urllib.request.urlopen(base + "tok", timeout=5).read() == b"tok.thumb"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "other", timeout=5)


def test_valid_certificate_is_not_renewed(tmp_path: Path) -> None:
    cert, key, _ = issue(make_ca(), "example.com", days=60)
    (tmp_path / "cert.pem").write_text(cert)
    (tmp_path / "key.pem").write_text(key)
    config = ACMEConfig(
        directory_url="https://unreachable.invalid/dir", domains=["example.com"], state_dir=str(tmp_path)
    )
    assert ensure_certificate(config) is False  # no network: nothing to do


def test_acme_settings_validation(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="not both"):
        ServerSettings(tls_acme_domains=("a.example",), tls_cert="c", data_dir=str(tmp_path))
    with pytest.raises(SettingsError, match="single-node"):
        ServerSettings(tls_acme_domains=("a.example",), peers=("p:1",), data_dir=str(tmp_path))
    with pytest.raises(SettingsError, match="state-dir or --data-dir"):
        ServerSettings(tls_acme_domains=("a.example",))


PEBBLE = os.environ.get("MEMBRANE_PEBBLE_DIRECTORY", "")


@pytest.mark.skipif(not PEBBLE, reason="set MEMBRANE_PEBBLE_DIRECTORY (and MEMBRANE_PEBBLE_CA) to run against Pebble")
def test_serve_with_an_acme_certificate_from_pebble(tmp_path: Path) -> None:
    domain = os.environ.get("MEMBRANE_PEBBLE_DOMAIN", "host.docker.internal")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    keys = tmp_path / "keys"
    keys.write_text("k:svc:read\n")
    keys.chmod(0o600)
    settings = ServerSettings(
        host="127.0.0.1",
        port=port,
        api_key_file=str(keys),
        tls_acme_domains=(domain,),
        tls_acme_directory=PEBBLE,
        tls_acme_ca_bundle=os.environ["MEMBRANE_PEBBLE_CA"],
        tls_acme_state_dir=str(tmp_path / "acme"),
        tls_acme_http_port=int(os.environ.get("MEMBRANE_PEBBLE_HTTP_PORT", "5002")),
        load_hooks=False,
    )
    server, mode = build_server(settings)
    assert mode == "API key over TLS"
    server.start()
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # Pebble's issuing root is generated per run
        deadline = time.monotonic() + 15
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5) as raw, context.wrap_socket(raw) as tls:
                    der = tls.getpeercert(binary_form=True)
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.1)
        served = x509.load_der_x509_certificate(der)
        names = served.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
            x509.DNSName
        )
        assert names == [domain]
    finally:
        server.stop(2.0)
