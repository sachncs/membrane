"""Regression tests: every non-probe route is authenticated when auth is on.

Covers the gaps fixed for production readiness:

* GET routes (``/retrieve``, ``/inventory``, ``/peers``, ``/metrics``,
  ``/metrics.json``) previously skipped the scope check entirely.
* Authentication failures escaped as unhandled exceptions (500)
  instead of 401 / 403.
* ``Server`` never attached an authenticator to the app.
* The mTLS CN was read from a client-controlled header.
* ``op_join`` let a CN register under another node's id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from membrane.auth.apikey import APIKeyAuthenticator, parse_keyfile
from membrane.auth.mtls import MTLSAuthenticator
from membrane.compute.cpu import CPU
from membrane.fragment import Fragment
from membrane.node import Node
from membrane.serialization import to_dict
from membrane.transfer import TransferService
from membrane.transport.fastapi import create_app
from membrane.transport.ops import op_join
from membrane.transport.tls import MTLSConfig
from membrane.transport.tls_protocol import peer_cn_from_transport, peer_headers_from_scope
from tests.conftest import make_fragment

KEYFILE = "reader-key:acme:read\nwriter-key:acme:read,write\nother-key:globex:read\n"


def _secret_fragment(content_hash: str = "secret") -> Fragment:
    base = make_fragment(content_hash)
    return Fragment(
        identity=base.identity,
        payload_ref=None,
        payload_size=base.payload_size,
        ttl=base.ttl,
        reuse_score=base.reuse_score,
        version_id=base.version_id,
        tenant_id="acme",
    )


@pytest.fixture
def client() -> TestClient:
    node = Node("n1", max_memory_bytes=10_000)
    assert node.store(_secret_fragment(), is_primary=True)
    app = create_app(
        node=node,
        compute_backend=CPU(),
        transfer_service=TransferService(),
        cluster_manager=None,
        authenticator=APIKeyAuthenticator(KEYFILE),
    )
    return TestClient(app, raise_server_exceptions=False)


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.parametrize(
    "path",
    ["/retrieve?content_hash=secret", "/inventory", "/peers", "/metrics", "/metrics.json", "/heartbeat"],
)
def test_reads_require_credentials(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert "secret" not in resp.text


def test_bad_key_is_401(client: TestClient) -> None:
    assert client.get("/inventory", headers=_bearer("nope")).status_code == 401


def test_probes_stay_public(client: TestClient) -> None:
    assert client.get("/livez").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_owner_tenant_can_read(client: TestClient) -> None:
    resp = client.get("/retrieve?content_hash=secret", headers=_bearer("reader-key"))
    assert resp.status_code == 200
    assert resp.json()["found"] is True


def test_other_tenant_cannot_read(client: TestClient) -> None:
    resp = client.get("/retrieve?content_hash=secret", headers=_bearer("other-key"))
    assert resp.status_code == 200
    assert resp.json()["found"] is False


def test_missing_scope_is_403(client: TestClient) -> None:
    body = {"fragment": to_dict(_secret_fragment("w1")), "is_primary": True}
    assert client.post("/store", json=body, headers=_bearer("reader-key")).status_code == 403
    assert client.post("/store", json=body, headers=_bearer("writer-key")).status_code == 200


def test_admin_routes_require_admin(client: TestClient) -> None:
    assert client.get("/admin/policy").status_code == 401
    assert client.get("/admin/policy", headers=_bearer("writer-key")).status_code == 403


def test_keyfile_rejects_empty_subject() -> None:
    """An empty subject would read as "unauthenticated" and skip the tenant check."""
    assert parse_keyfile("k1::read\nk2:svc:read\n").keys() == {"k2"}


# ---------------------------------------------------------------------------
# mTLS peer CN comes from the verified certificate, never a header
# ---------------------------------------------------------------------------


def _mtls_cfg() -> MTLSConfig:
    return MTLSConfig(
        server_cert_pem="CERT",
        server_key_pem="KEY",
        ca_bundle_pem="CA",
        allowed_cns=frozenset({"admin-1", "read-3"}),
    )


def test_client_supplied_cn_header_is_discarded() -> None:
    headers = peer_headers_from_scope({}, [("X-SSL-Client-CN", "admin-1"), ("Accept", "*/*")])
    assert "x-ssl-client-cn" not in headers
    assert headers["accept"] == "*/*"


def test_verified_cn_overrides_forged_header() -> None:
    scope = {"extensions": {"tls": {"peer_cn": "read-3"}}}
    headers = peer_headers_from_scope(scope, [("x-ssl-client-cn", "admin-1")])
    assert headers["x-ssl-client-cn"] == "read-3"


def test_forged_cn_header_cannot_authenticate_over_http() -> None:
    node = Node("n1", max_memory_bytes=10_000)
    app = create_app(
        node=node,
        compute_backend=CPU(),
        transfer_service=TransferService(),
        cluster_manager=None,
        authenticator=MTLSAuthenticator(_mtls_cfg()),
    )
    resp = TestClient(app).get("/inventory", headers={"X-SSL-Client-CN": "admin-1"})
    assert resp.status_code == 401


def test_peer_cn_from_transport_reads_verified_subject() -> None:
    cert = {"subject": ((("organizationName", "acme"),), (("commonName", "write-2"),))}
    ssl_object = SimpleNamespace(getpeercert=lambda: cert)
    transport = SimpleNamespace(get_extra_info=lambda name: ssl_object if name == "ssl_object" else None)
    assert peer_cn_from_transport(transport) == "write-2"  # type: ignore[arg-type]
    plaintext = SimpleNamespace(get_extra_info=lambda name: None)
    assert peer_cn_from_transport(plaintext) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("cn", "node_id", "admitted"),
    [("admin-1", "admin-1", True), ("admin-1", "1", True), ("admin-1", "admin-2", False), ("read-3", "n2", False)],
)
def test_join_cn_must_match_node_id(cn: str, node_id: str, admitted: bool) -> None:
    cluster = MagicMock()
    cluster.membership.to_json.return_value = []
    cfg = MTLSConfig(server_cert_pem="CERT", server_key_pem="KEY", ca_bundle_pem="CA", allowed_cns=frozenset({cn}))
    status, _ = op_join(
        cluster, node_id, "10.0.0.2", 8080, headers={"x-ssl-client-cn": cn}, authenticator=MTLSAuthenticator(cfg)
    )
    assert (status == 200) is admitted


def test_join_response_includes_seed() -> None:
    cluster = MagicMock()
    cluster.membership.to_json.return_value = []
    cluster.node_id = "seed"
    cluster.advertise_host = "seed.local"
    cluster.port = 8080
    status, body = op_join(cluster, "n2", "10.0.0.2", 8080)
    assert status == 200
    assert {"node_id": "seed", "host": "seed.local", "port": 8080} in body["peers"]


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


def test_server_attaches_authenticator() -> None:
    from membrane.server import Server

    auth = APIKeyAuthenticator(KEYFILE)
    server = Server(node=Node("n1", max_memory_bytes=10_000), authenticator=auth, host="127.0.0.1", port=0)
    assert server.transport.app.state.authenticator is auth
    assert server.transport.app.state.server is server


def test_server_builds_mtls_authenticator_from_tls() -> None:
    from membrane.server import Server

    server = Server(node=Node("n1", max_memory_bytes=10_000), tls=_mtls_cfg(), host="127.0.0.1", port=0)
    assert isinstance(server.transport.app.state.authenticator, MTLSAuthenticator)


# ---------------------------------------------------------------------------
# CLI: fail closed
# ---------------------------------------------------------------------------


def test_serve_refuses_unauthenticated_public_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    from membrane.cli import app

    for var in ("MEMBRANE_API_KEY_FILE", "MEMBRANE_ALLOW_UNAUTHENTICATED", "MEMBRANE_TLS_CERT_FILE"):
        monkeypatch.delenv(var, raising=False)
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0", "--daemon"])
    assert result.exit_code == 2
    assert "Refusing to serve unauthenticated" in result.output


def test_serve_rejects_empty_keyfile(tmp_path) -> None:
    from membrane.cli import app

    keyfile = tmp_path / "keys"
    keyfile.write_text("# no keys\n")
    result = CliRunner().invoke(app, ["serve", "--daemon", "--api-key-file", str(keyfile)])
    assert result.exit_code == 2
    assert "no valid keys" in result.output


def test_prefill_fragments_belong_to_caller_tenant(client: TestClient) -> None:
    resp = client.post("/prefill", json={"prompt_tokens": [7, 8, 9], "model_id": "m"}, headers=_bearer("writer-key"))
    frag = resp.json()["fragments"][0]
    assert frag["tenant_id"] == "acme"
    content_hash = frag["identity"]["payload_hash"]
    assert client.get(f"/retrieve?content_hash={content_hash}", headers=_bearer("other-key")).json()["found"] is False
