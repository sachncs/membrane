"""ACMEClient against a minimal in-process ACME CA (RFC 8555, HTTP-01).

The CA checks every JWS nonce, fetches the key authorization from the
client's HTTP-01 responder before validating, and signs the CSR with a
test CA. Pebble covers the same flow in CI's ACME job.
"""

import base64
import datetime
import hashlib
import http.server
import json
import socket
import threading
import urllib.request
from collections.abc import Iterator

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from membrane.transport.acme import (
    CHALLENGE_PREFIX,
    ACMEClient,
    ACMEConfig,
    ACMEError,
    certificate_valid_for,
    ensure_certificate,
)
from tests.tls_helpers import make_ca


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeCA:
    """Just enough of an ACME server for one account and its orders."""

    def __init__(self, http_port: int, challenge_types: tuple[str, ...] = ("http-01",)) -> None:
        self.http_port = http_port
        self.challenge_types = challenge_types
        self.ca = make_ca()
        self.nonces: set[str] = set()
        self.counter = 0
        self.thumbprint = ""
        self.authz_status = "pending"
        self.order_status = "pending"
        self.certificate = ""
        self.bad_nonce_once = True
        self.refuse_orders = False
        self.stall = False
        self.domains: list[str] = []
        ca = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def reply(self, status: int, body: object = None, headers: dict[str, str] | None = None) -> None:
                data = body.encode() if isinstance(body, str) else json.dumps(body or {}).encode()
                self.send_response(status)
                content_type = "application/pem-certificate-chain" if isinstance(body, str) else "application/json"
                self.send_header("Content-Type", content_type if status < 400 else "application/problem+json")
                self.send_header("Replay-Nonce", ca.new_nonce())
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_HEAD(self) -> None:
                self.reply(200)

            def do_GET(self) -> None:
                base = ca.base
                self.reply(
                    200, {"newNonce": f"{base}/nonce", "newAccount": f"{base}/account", "newOrder": f"{base}/order"}
                )

            def do_POST(self) -> None:
                jws = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                protected = json.loads(unb64(jws["protected"]))
                payload = json.loads(unb64(jws["payload"])) if jws["payload"] else None
                if ca.bad_nonce_once:
                    ca.bad_nonce_once = False
                    return self.reply(400, {"type": "urn:ietf:params:acme:error:badNonce", "detail": "stale"})
                assert protected["nonce"] in ca.nonces, "replayed nonce"
                ca.nonces.discard(protected["nonce"])
                assert protected["url"] == ca.base + self.path
                self.route(protected, payload)

            def route(self, protected: dict, payload: object) -> None:
                base, path = ca.base, self.path
                if path == "/account":
                    jwk = protected["jwk"]
                    canonical = json.dumps(jwk, sort_keys=True, separators=(",", ":")).encode()
                    ca.thumbprint = base64.urlsafe_b64encode(hashlib.sha256(canonical).digest()).rstrip(b"=").decode()
                    return self.reply(201, {"status": "valid"}, {"Location": f"{base}/acct/1"})
                assert protected["kid"] == f"{base}/acct/1"
                if path == "/order":
                    if ca.refuse_orders:
                        return self.reply(
                            403, {"type": "urn:ietf:params:acme:error:rejectedIdentifier", "detail": "no"}
                        )
                    ca.domains = [i["value"] for i in payload["identifiers"]]
                    return self.reply(201, ca.order(), {"Location": f"{base}/order/1"})
                if path == "/order/1":
                    return self.reply(200, ca.order())
                if path == "/authz/1":
                    challenges = [{"type": t, "url": f"{base}/chall/1", "token": "tok-1"} for t in ca.challenge_types]
                    return self.reply(
                        200,
                        {"status": ca.authz_status, "identifier": {"value": ca.domains[0]}, "challenges": challenges},
                    )
                if path == "/chall/1":
                    ca.validate()
                    return self.reply(200, {"status": "processing"})
                if path == "/finalize/1":
                    ca.sign(x509.load_der_x509_csr(unb64(payload["csr"])))
                    return self.reply(200, ca.order())
                if path == "/cert/1":
                    return self.reply(200, ca.certificate)
                return self.reply(404, {"detail": path})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def new_nonce(self) -> str:
        self.counter += 1
        nonce = f"n{self.counter}"
        self.nonces.add(nonce)
        return nonce

    def order(self) -> dict:
        body = {
            "status": self.order_status,
            "authorizations": [f"{self.base}/authz/1"],
            "finalize": f"{self.base}/finalize/1",
        }
        if self.order_status == "valid":
            body["certificate"] = f"{self.base}/cert/1"
        return body

    def validate(self) -> None:
        """Fetch the key authorization from the client's responder, as a CA does."""
        if self.stall:
            return
        url = f"http://127.0.0.1:{self.http_port}{CHALLENGE_PREFIX}tok-1"
        with urllib.request.urlopen(url, timeout=5) as response:
            answer = response.read().decode()
        self.authz_status = "valid" if answer == f"tok-1.{self.thumbprint}" else "invalid"

    def sign(self, csr: x509.CertificateSigningRequest) -> None:
        ca_cert, ca_key = self.ca
        now = datetime.datetime.now(datetime.UTC)
        names = csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        cert = (
            x509.CertificateBuilder()
            .subject_name(csr.subject)
            .issuer_name(ca_cert.subject)
            .public_key(csr.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=90))
            .add_extension(names, critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        self.certificate = (
            cert.public_bytes(serialization.Encoding.PEM) + ca_cert.public_bytes(serialization.Encoding.PEM)
        ).decode()
        self.order_status = "valid"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ca() -> Iterator[FakeCA]:
    authority = FakeCA(free_port())
    yield authority
    authority.close()


def config_for(ca: FakeCA, tmp_path, **overrides) -> ACMEConfig:
    values = {
        "directory_url": ca.base,
        "domains": ["node.example", "alt.example"],
        "state_dir": str(tmp_path / "acme"),
        "contact": ["mailto:ops@example.com"],
        "http_port": ca.http_port,
        "poll_interval_sec": 0.01,
        "timeout_sec": 5.0,
    }
    values.update(overrides)
    return ACMEConfig(**values)


def test_issues_renews_only_when_due_and_keeps_the_account(ca: FakeCA, tmp_path) -> None:
    config = config_for(ca, tmp_path)
    assert certificate_valid_for(config.cert_path) is None
    assert ensure_certificate(config) is True
    cert = x509.load_pem_x509_certificate(config.cert_path.read_bytes())
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert names.get_values_for_type(x509.DNSName) == ["node.example", "alt.example"]
    assert config.key_path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "acme" / "account.key").stat().st_mode & 0o777 == 0o600
    remaining = certificate_valid_for(config.cert_path)
    assert remaining is not None and remaining > datetime.timedelta(days=89)
    # Still valid for longer than the renewal window: nothing to do.
    assert ensure_certificate(config) is False
    # Inside the window: renew, reusing the stored account key.
    assert ensure_certificate(config, renew_before_days=100) is True


def test_a_wrong_key_authorization_fails_the_order(ca: FakeCA, tmp_path) -> None:
    original = ca.validate

    def tamper() -> None:
        ca.thumbprint = "someone-else"
        original()

    ca.validate = tamper  # type: ignore[method-assign]
    client = ACMEClient(config_for(ca, tmp_path))
    with pytest.raises(ACMEError, match="became invalid"):
        client.issue()
    client.close()


def test_an_authorization_that_never_completes_times_out(ca: FakeCA, tmp_path) -> None:
    ca.stall = True
    client = ACMEClient(config_for(ca, tmp_path, timeout_sec=0.2))
    with pytest.raises(ACMEError, match="still pending"):
        client.issue()
    client.close()


def test_error_responses_and_missing_inputs(ca: FakeCA, tmp_path) -> None:
    ca.refuse_orders = True
    client = ACMEClient(config_for(ca, tmp_path))
    with pytest.raises(ACMEError, match=r"failed \(403\): no"):
        client.issue()
    with pytest.raises(ACMEError, match="no domains"):
        ACMEClient(config_for(ca, tmp_path, domains=[])).issue()
    client.close()


def test_only_http01_is_supported(tmp_path) -> None:
    authority = FakeCA(free_port(), challenge_types=("dns-01",))
    try:
        client = ACMEClient(config_for(authority, tmp_path))
        with pytest.raises(ACMEError, match=r"no http-01 challenge offered for node\.example"):
            client.issue()
        client.close()
    finally:
        authority.close()


def test_unreadable_certificates_count_as_missing(tmp_path) -> None:
    garbage = tmp_path / "cert.pem"
    garbage.write_text("not a certificate")
    assert certificate_valid_for(garbage) is None
