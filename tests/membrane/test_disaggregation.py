"""Disaggregated prefill / decode on running nodes: REST, gRPC, roles, tenants, KV bundles."""

import json
from typing import override
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from membrane.auth.apikey import APIKeyAuthenticator
from membrane.compute.cpu import CPU
from membrane.disagg.grpc import (
    build_decode_request_message,
    build_prefill_request_message,
    decode_response_from_message,
    make_channel,
    make_stub,
    response_from_message,
    serve,
)
from membrane.disagg.protocol import DecodeRequest, PrefillRequest
from membrane.network.membership import Membership
from membrane.network.peer import Peer, RawResponse
from membrane.node import Node
from membrane.registry import Registry
from membrane.ring import Ring
from membrane.runtime.settings import ServerSettings, SettingsError, grpc_supported
from membrane.server import Server
from membrane.services import ServiceOptions
from membrane.shard import Shard

KEYS = "alice-key:alice:read,write\nbob-key:bob:read,write\nreader-key:reader:read\n"


class EchoBackend(CPU):
    """CPU simulator whose decode echoes the prompt's last tokens."""

    @override
    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        return {"text": "", "tokens": list(reversed(prompt_tokens))[:max_tokens]}


def make_node(name: str, role: str = "both", **options) -> tuple[Server, TestClient]:
    server = Server(
        node=Node(name), compute=EchoBackend(), port=0, load_hooks=False, services=ServiceOptions(role=role), **options
    )
    return server, TestClient(server.transport.app)


def prefill_body(tokens: list[int], request_id: str = "r1", model_id: str = "m") -> dict:
    return PrefillRequest(request_id=request_id, model_id=model_id, token_ids=tuple(tokens)).to_dict()


def decode_body(handle: str, model_id: str = "m", max_tokens: int = 4) -> dict:
    return DecodeRequest(request_id="d1", kv_handle=handle, model_id=model_id, max_tokens=max_tokens).to_dict()


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_prefill_then_decode_on_one_node() -> None:
    server, client = make_node("pd")
    tokens = list(range(300))
    prefilled = client.post("/disagg/prefill", json=prefill_body(tokens)).json()
    assert prefilled["prompt_len"] == 300
    assert prefilled["cached_prefix_len"] == 0
    assert len(server.node.fragments) == 4  # three KV windows and the manifest
    again = client.post("/disagg/prefill", json=prefill_body(tokens, "r2")).json()
    assert again["cached_prefix_len"] == 300
    assert again["kv_handle"] == prefilled["kv_handle"]
    decoded = client.post("/disagg/decode", json=decode_body(prefilled["kv_handle"])).json()
    assert decoded == {"request_id": "d1", "token_ids": [299, 298, 297, 296], "finished": True}
    batch = client.post("/disagg/prefill/batch", json={"requests": [prefill_body([1, 2, 3], "b1")]}).json()
    assert batch["responses"][0]["prompt_len"] == 3
    assert client.get("/disagg/healthz").json() == {"status": "ok"}


def test_decode_errors() -> None:
    _server, client = make_node("pd")
    assert client.post("/disagg/decode", json=decode_body("0" * 64)).status_code == 404
    handle = client.post("/disagg/prefill", json=prefill_body([1, 2, 3])).json()["kv_handle"]
    assert client.post("/disagg/decode", json=decode_body(handle, model_id="other")).status_code == 400
    assert client.post("/disagg/prefill", json={"bad": 1}).status_code == 400


def test_roles_gate_each_phase() -> None:
    _prefill, prefill_client = make_node("p", role="prefill")
    _decode, decode_client = make_node("d", role="decode")
    handle = prefill_client.post("/disagg/prefill", json=prefill_body([1, 2])).json()["kv_handle"]
    assert prefill_client.post("/disagg/decode", json=decode_body(handle)).status_code == 409
    assert decode_client.post("/disagg/prefill", json=prefill_body([1, 2])).status_code == 409
    assert prefill_client.get("/heartbeat").json()["role"] == "prefill"


class AppTransport:
    """Peer transport that calls an in-process app."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(self, method, url, body, headers, timeout_sec):
        parts = urlsplit(url)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        response = self.client.request(method, path, content=body, headers=headers)
        return json.loads(response.text) if response.text else {}

    def request_bytes(self, method, url, body, headers, timeout_sec):
        response = self.client.request(method, urlsplit(url).path, headers=headers)
        return RawResponse(response.status_code, {k.lower(): v for k, v in response.headers.items()}, response.content)


class FakeCluster:
    """The parts of a cluster manager the services read."""

    def __init__(self, local: str) -> None:
        self.hash_ring = Ring()
        self.hash_ring.add_node(local)
        self.directory = Registry()
        self.membership = Membership(local, self.hash_ring, Shard(self.hash_ring), self.directory)


def test_decode_node_pulls_kv_from_the_prefill_node() -> None:
    prefill, prefill_client = make_node("p", role="prefill")
    decode, decode_client = make_node("d", role="decode")
    cluster = FakeCluster("d")
    cluster.membership.add("p", "127.0.0.1", 9001)
    cluster.membership.record_heartbeat("p", report={"role": "prefill"})
    cluster.membership.clients["p"] = Peer(
        "http://p:9001", transport=AppTransport(prefill_client), max_retries=1, retry_delay_sec=0.0
    )
    decode.services.view.cluster = cluster

    tokens = list(range(200))
    handle = prefill_client.post("/disagg/prefill", json=prefill_body(tokens)).json()["kv_handle"]
    decoded = decode_client.post("/disagg/decode", json=decode_body(handle, max_tokens=2)).json()
    assert decoded["token_ids"] == [199, 198]
    kv = [h for h, f in prefill.node.fragments.items() if f.identity.model_id == "m"]
    assert sorted(h for h in decode.node.fragments if h in kv) == sorted(kv)
    assert decode.services.disagg.fetched == 2
    assert not set(kv) & decode.node.primary_hashes  # copies, not owners
    # The second decode finds everything locally.
    decode_client.post("/disagg/decode", json=decode_body(handle))
    assert decode.services.disagg.fetched == 2


def test_disagg_routes_authenticate_and_isolate_tenants() -> None:
    _server, client = make_node("pd", authenticator=APIKeyAuthenticator(keyfile_text=KEYS))
    body = prefill_body([5, 6, 7])
    assert client.post("/disagg/prefill", json=body).status_code == 401
    assert client.post("/disagg/prefill", json=body, headers=auth("reader-key")).status_code == 403
    assert client.get("/disagg/healthz").status_code == 200
    handle = client.post("/disagg/prefill", json=body, headers=auth("alice-key")).json()["kv_handle"]
    assert client.post("/disagg/decode", json=decode_body(handle), headers=auth("alice-key")).status_code == 200
    # Same prompt, another tenant: the handle is not theirs.
    assert client.post("/disagg/decode", json=decode_body(handle), headers=auth("bob-key")).status_code == 404


def test_grpc_surface_authenticates_and_serves() -> None:
    grpc = pytest.importorskip("grpc")

    server, _client = make_node("g", authenticator=APIKeyAuthenticator(keyfile_text=KEYS))
    disagg = server.services.disagg
    grpc_server, port = serve(
        disagg.prefill_service, disagg.decode_service, "127.0.0.1", 0, authenticator=server.authenticator
    )
    try:
        stub = make_stub(make_channel(f"127.0.0.1:{port}"))
        request = build_prefill_request_message(PrefillRequest(request_id="g1", model_id="m", token_ids=(1, 2, 3)))
        with pytest.raises(grpc.RpcError) as missing:
            stub.Prefill(request)
        assert missing.value.code() == grpc.StatusCode.UNAUTHENTICATED
        with pytest.raises(grpc.RpcError) as forbidden:
            stub.Prefill(request, metadata=[("authorization", "Bearer reader-key")])
        assert forbidden.value.code() == grpc.StatusCode.PERMISSION_DENIED
        metadata = [("authorization", "Bearer alice-key")]
        prefilled = response_from_message(stub.Prefill(request, metadata=metadata))
        decode = DecodeRequest(request_id="g2", kv_handle=prefilled.kv_handle, model_id="m", max_tokens=2)
        decoded = decode_response_from_message(stub.Decode(build_decode_request_message(decode), metadata=metadata))
        assert decoded.token_ids == (3, 2)
        unknown = DecodeRequest(request_id="g3", kv_handle="0" * 64, model_id="m")
        with pytest.raises(grpc.RpcError) as not_found:
            stub.Decode(build_decode_request_message(unknown), metadata=metadata)
        assert not_found.value.code() == grpc.StatusCode.NOT_FOUND
        # Bob cannot decode Alice's handle over gRPC either.
        with pytest.raises(grpc.RpcError) as other_tenant:
            stub.Decode(build_decode_request_message(decode), metadata=[("authorization", "Bearer bob-key")])
        assert other_tenant.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        grpc_server.stop(grace=0)


def test_server_starts_grpc_on_its_port() -> None:
    grpc = pytest.importorskip("grpc")

    server = Server(node=Node("gs"), port=0, load_hooks=False, grpc_port=0, services=ServiceOptions(role="prefill"))
    server.start()
    try:
        assert server.grpc_port
        stub = make_stub(make_channel(f"127.0.0.1:{server.grpc_port}"))
        decode = DecodeRequest(request_id="x", kv_handle="0" * 64, model_id="m")
        with pytest.raises(grpc.RpcError) as wrong_role:
            stub.Decode(build_decode_request_message(decode))
        assert wrong_role.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.stop()


def test_disagg_settings() -> None:
    with pytest.raises(SettingsError, match="role"):
        ServerSettings(role="mixer")
    with pytest.raises(SettingsError, match="gRPC port"):
        ServerSettings(grpc_port=70000)
    if grpc_supported():
        assert ServerSettings(role="decode", grpc_port=9090).grpc_port == 9090
    else:  # no grpcio, or a free-threaded Python where it would re-enable the GIL
        with pytest.raises(SettingsError, match="grpcio"):
            ServerSettings(role="decode", grpc_port=9090)


def test_kv_bundles_are_named_and_tenant_scoped() -> None:
    _server, client = make_node("kv", authenticator=APIKeyAuthenticator(keyfile_text=KEYS))
    put = client.put("/kv/req-1.0?model_id=m", content=b"kv-bytes", headers=auth("alice-key"))
    assert put.status_code == 200
    assert client.get("/kv/req-1.0?model_id=m", headers=auth("alice-key")).content == b"kv-bytes"
    assert client.head("/kv/req-1.0?model_id=m", headers=auth("alice-key")).status_code == 200
    assert client.get("/kv/req-1.0?model_id=m", headers=auth("bob-key")).status_code == 404
    assert client.get("/kv/req-1.0?model_id=other", headers=auth("alice-key")).status_code == 404
    client.put("/kv/req-1.0?model_id=m", content=b"newer", headers=auth("alice-key"))
    assert client.get("/kv/req-1.0?model_id=m", headers=auth("alice-key")).content == b"newer"
    assert client.put("/kv/bad%20name?model_id=m", content=b"x", headers=auth("alice-key")).status_code == 400
    assert client.put("/kv/req-2?model_id=m", content=b"x", headers=auth("reader-key")).status_code == 403
