"""ACME (RFC 8555) certificates for a public listener, with HTTP-01 challenges.

``membrane serve --tls-acme-domain example.com`` obtains a certificate
from an ACME CA (Let's Encrypt by default) before the listener starts,
keeps it in ``--tls-acme-state-dir``, and renews it in the background
when it has less than :data:`RENEW_BEFORE_DAYS` left. A renewed
certificate is served without a restart.

HTTP-01 means the CA fetches
``http://<domain>/.well-known/acme-challenge/<token>``: port 80 on the
domain must reach :class:`ChallengeResponder`, which listens on
``--tls-acme-http-port`` only while an order is open.

The account key and the certificate key are ECDSA P-256; requests are
JWS-signed with ES256 (RFC 7515/7518).
"""

import base64
import datetime
import hashlib
import http.server
import json
import logging
import os
import ssl
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

logger = logging.getLogger(__name__)

LETS_ENCRYPT = "https://acme-v02.api.letsencrypt.org/directory"
RENEW_BEFORE_DAYS = 30
CHALLENGE_PREFIX = "/.well-known/acme-challenge/"


class ACMEError(RuntimeError):
    """An ACME request failed or an order did not become valid."""


@dataclass
class ACMEConfig:
    """Where and how to obtain a certificate.

    Attributes:
        directory_url: ACME directory URL.
        domains: DNS names the certificate covers.
        state_dir: Directory holding ``account.key``, ``cert.pem``, and
            ``key.pem``.
        contact: Contact URIs (e.g. ``["mailto:ops@example.com"]``).
        http_port: Port the HTTP-01 responder listens on.
        ca_bundle: CA bundle trusted for the directory's HTTPS (private
            or test CAs); the system store when empty.
        poll_interval_sec: Delay between order and authorization polls.
        timeout_sec: Budget for one issuance.
    """

    directory_url: str = LETS_ENCRYPT
    domains: list[str] = field(default_factory=list)
    state_dir: str = ""
    contact: list[str] | None = None
    http_port: int = 80
    ca_bundle: str = ""
    poll_interval_sec: float = 1.0
    timeout_sec: float = 120.0

    @property
    def cert_path(self) -> Path:
        """The certificate chain file."""
        return Path(self.state_dir) / "cert.pem"

    @property
    def key_path(self) -> Path:
        """The certificate's private key file."""
        return Path(self.state_dir) / "key.pem"


def b64url(data: bytes) -> str:
    """Base64url-encode without padding (RFC 7515).

    Args:
        data: Bytes to encode.

    Returns:
        str: The encoding.
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def write_private(path: Path, data: bytes) -> None:
    """Atomically write ``data`` with mode 0600.

    Args:
        path: Destination.
        data: Contents.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


class ChallengeResponder:
    """Serves HTTP-01 key authorizations while an order is open.

    Attributes:
        port: Listening port.
        tokens: ``token -> key authorization``.
    """

    def __init__(self, port: int, host: str = "0.0.0.0") -> None:
        """Create a stopped responder.

        Args:
            port: Listening port (``0`` picks a free one).
            host: Bind address.
        """
        self.port = port
        self.host = host
        self.tokens: dict[str, str] = {}
        self.__server: http.server.ThreadingHTTPServer | None = None

    def __enter__(self) -> ChallengeResponder:
        """Start listening.

        Returns:
            ChallengeResponder: This responder.
        """
        tokens = self.tokens

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                token = self.path.removeprefix(CHALLENGE_PREFIX)
                answer = tokens.get(token) if self.path.startswith(CHALLENGE_PREFIX) else None
                body = (answer or "not found").encode()
                self.send_response(200 if answer else 404)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            @override
            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("acme responder: " + format, *args)

        self.__server = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self.__server.server_address[1]
        threading.Thread(target=self.__server.serve_forever, daemon=True, name="membrane-acme-http01").start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop listening.

        Args:
            *_exc: Exception details (ignored).
        """
        if self.__server is not None:
            self.__server.shutdown()
            self.__server.server_close()
            self.__server = None


class ACMEClient:
    """RFC 8555 client: account, order, HTTP-01 authorization, finalize, download."""

    def __init__(self, config: ACMEConfig) -> None:
        """Create a client; the account key is loaded or created on first use.

        Args:
            config: Directory, domains, and state location.
        """
        import httpx

        self.config = config
        verify: Any = ssl.create_default_context(cafile=config.ca_bundle) if config.ca_bundle else True
        self.__http = httpx.Client(timeout=30.0, verify=verify, follow_redirects=False)
        self.__directory: dict[str, Any] = {}
        self.__nonce = ""
        self.__kid = ""
        self.__account_key: Any = None

    # -- keys and JWS -------------------------------------------------

    def account_key(self) -> Any:
        """Load or create the account key (``state_dir/account.key``, mode 0600).

        Returns:
            Any: An ECDSA P-256 private key.
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        if self.__account_key is None:
            path = Path(self.config.state_dir) / "account.key"
            if path.exists():
                self.__account_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            else:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self.__account_key = ec.generate_private_key(ec.SECP256R1())
                write_private(
                    path,
                    self.__account_key.private_bytes(
                        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
                    ),
                )
        return self.__account_key

    def jwk(self) -> dict[str, str]:
        """Return the account public key as a JWK.

        Returns:
            dict[str, str]: ``{"crv", "kty", "x", "y"}``.
        """
        numbers = self.account_key().public_key().public_numbers()
        return {
            "crv": "P-256",
            "kty": "EC",
            "x": b64url(numbers.x.to_bytes(32, "big")),
            "y": b64url(numbers.y.to_bytes(32, "big")),
        }

    def thumbprint(self) -> str:
        """Return the RFC 7638 JWK thumbprint used in key authorizations.

        Returns:
            str: Base64url SHA-256 of the canonical JWK.
        """
        canonical = json.dumps(self.jwk(), sort_keys=True, separators=(",", ":")).encode()
        return b64url(hashlib.sha256(canonical).digest())

    def sign(self, url: str, payload: Any) -> bytes:
        """Build a flattened JWS for ``url``.

        Args:
            url: Request URL (signed into the protected header).
            payload: JSON payload, or ``None`` for POST-as-GET.

        Returns:
            bytes: The JWS JSON body.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

        protected: dict[str, Any] = {"alg": "ES256", "nonce": self.nonce(), "url": url}
        if self.__kid:
            protected["kid"] = self.__kid
        else:
            protected["jwk"] = self.jwk()
        protected_b64 = b64url(json.dumps(protected).encode())
        payload_b64 = "" if payload is None else b64url(json.dumps(payload).encode())
        der = self.account_key().sign(f"{protected_b64}.{payload_b64}".encode(), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return json.dumps({"protected": protected_b64, "payload": payload_b64, "signature": b64url(signature)}).encode()

    # -- protocol -----------------------------------------------------

    def directory(self) -> dict[str, Any]:
        """Fetch (once) the ACME directory.

        Returns:
            dict[str, Any]: Endpoint URLs.
        """
        if not self.__directory:
            response = self.__http.get(self.config.directory_url)
            response.raise_for_status()
            self.__directory = response.json()
        return self.__directory

    def nonce(self) -> str:
        """Return a fresh replay nonce.

        Returns:
            str: The nonce from the last response, or a new one.
        """
        if self.__nonce:
            nonce, self.__nonce = self.__nonce, ""
            return nonce
        response = self.__http.head(self.directory()["newNonce"])
        return response.headers["Replay-Nonce"]

    def post(self, url: str, payload: Any) -> Any:
        """Send a signed POST, retrying once on ``badNonce``.

        Args:
            url: Request URL.
            payload: JSON payload, or ``None`` for POST-as-GET.

        Returns:
            Any: The ``httpx.Response``.

        Raises:
            ACMEError: On an error response.
        """
        for attempt in range(2):
            response = self.__http.post(
                url, content=self.sign(url, payload), headers={"Content-Type": "application/jose+json"}
            )
            self.__nonce = response.headers.get("Replay-Nonce", "")
            if response.status_code < 400:
                return response
            problem = response.json() if "json" in response.headers.get("Content-Type", "") else {}
            if problem.get("type", "").endswith(":badNonce") and attempt == 0:
                continue
            raise ACMEError(f"ACME {url} failed ({response.status_code}): {problem.get('detail', response.text)}")
        raise ACMEError(f"ACME {url} kept rejecting the nonce")  # pragma: no cover

    def register(self) -> str:
        """Create (or look up) the account and remember its URL.

        Returns:
            str: The account URL (JWS ``kid``).
        """
        if not self.__kid:
            payload: dict[str, Any] = {"termsOfServiceAgreed": True}
            if self.config.contact:
                payload["contact"] = self.config.contact
            response = self.post(self.directory()["newAccount"], payload)
            self.__kid = response.headers["Location"]
        return self.__kid

    def wait_for(self, url: str, done: set[str]) -> dict[str, Any]:
        """Poll a resource until its status is in ``done``.

        Args:
            url: Order or authorization URL.
            done: Terminal statuses to stop at.

        Returns:
            dict[str, Any]: The resource.

        Raises:
            ACMEError: When it becomes ``invalid`` or the timeout passes.
        """
        deadline = time.monotonic() + self.config.timeout_sec
        while True:
            resource = self.post(url, None).json()
            status = resource.get("status")
            if status in done:
                return resource
            if status == "invalid":
                raise ACMEError(f"ACME resource {url} became invalid: {json.dumps(resource)[:500]}")
            if time.monotonic() >= deadline:
                raise ACMEError(f"ACME resource {url} still {status} after {self.config.timeout_sec}s")
            time.sleep(self.config.poll_interval_sec)

    def issue(self) -> tuple[str, str]:
        """Order, authorize (HTTP-01), finalize, and download a certificate.

        Returns:
            tuple[str, str]: ``(certificate chain PEM, private key PEM)``.

        Raises:
            ACMEError: When any step fails.
        """
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        if not self.config.domains:
            raise ACMEError("no domains to certify")
        self.register()
        order_response = self.post(
            self.directory()["newOrder"], {"identifiers": [{"type": "dns", "value": d} for d in self.config.domains]}
        )
        order_url = order_response.headers["Location"]
        order = order_response.json()
        with ChallengeResponder(self.config.http_port) as responder:
            for authorization_url in order["authorizations"]:
                authorization = self.post(authorization_url, None).json()
                if authorization.get("status") == "valid":
                    continue
                challenge = next((c for c in authorization["challenges"] if c["type"] == "http-01"), None)
                if challenge is None:
                    raise ACMEError(f"no http-01 challenge offered for {authorization['identifier']['value']}")
                responder.tokens[challenge["token"]] = f"{challenge['token']}.{self.thumbprint()}"
                self.post(challenge["url"], {})
                self.wait_for(authorization_url, {"valid"})
        key = ec.generate_private_key(ec.SECP256R1())
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, self.config.domains[0])]))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(d) for d in self.config.domains]), critical=False)
            .sign(key, hashes.SHA256())
        )
        self.post(order["finalize"], {"csr": b64url(csr.public_bytes(serialization.Encoding.DER))})
        order = self.wait_for(order_url, {"valid"})
        chain = self.post(order["certificate"], None).text
        key_pem = key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ).decode()
        return chain, key_pem

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self.__http.close()


def certificate_valid_for(path: Path) -> datetime.timedelta | None:
    """Return how long the certificate at ``path`` stays valid.

    Args:
        path: Certificate PEM file.

    Returns:
        datetime.timedelta | None: Time until ``notAfter``, or ``None`` when
        the file is missing or unreadable.
    """
    from membrane.transport.tls_rotation import cert_not_after

    try:
        expires = cert_not_after(path.read_text())
    except OSError:
        return None
    if expires is None:
        return None
    return expires - datetime.datetime.now(datetime.UTC)


def ensure_certificate(config: ACMEConfig, renew_before_days: float = RENEW_BEFORE_DAYS) -> bool:
    """Make sure ``state_dir`` holds a certificate valid for at least ``renew_before_days``.

    Args:
        config: ACME configuration.
        renew_before_days: Renewal window.

    Returns:
        bool: True when a new certificate was issued.

    Raises:
        ACMEError: When issuance fails.
    """
    remaining = certificate_valid_for(config.cert_path)
    if remaining is not None and remaining > datetime.timedelta(days=renew_before_days):
        return False
    client = ACMEClient(config)
    try:
        chain, key = client.issue()
    finally:
        client.close()
    Path(config.state_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(config.key_path, key.encode())
    write_private(config.cert_path, chain.encode())
    logger.info("ACME certificate issued for %s", ", ".join(config.domains))
    return True


__all__ = [
    "CHALLENGE_PREFIX",
    "LETS_ENCRYPT",
    "RENEW_BEFORE_DAYS",
    "ACMEClient",
    "ACMEConfig",
    "ACMEError",
    "ChallengeResponder",
    "b64url",
    "certificate_valid_for",
    "ensure_certificate",
    "write_private",
]
