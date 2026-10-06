"""The memory API a node serves: reconstruct, prefix lookup, sessions, typed objects, compat."""

import base64

import pytest
from fastapi.testclient import TestClient

from membrane.auth.apikey import APIKeyAuthenticator
from membrane.compute.hashing import token_hash
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.node import Node
from membrane.serialization import to_dict
from membrane.server import Server
from membrane.services import ServiceOptions

KEYS = "alice-key:alice:read,write\nbob-key:bob:read,write\nroot-key:root:admin\n"


def make_server(**options) -> tuple[Server, TestClient]:
    server = Server(node=Node("mem", max_memory_bytes=1 << 26), port=0, load_hooks=False, **options)
    return server, TestClient(server.transport.app)


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def prompt(length: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(length)]


def test_reconstruct_returns_prefilled_windows() -> None:
    _server, client = make_server()
    tokens = prompt(300)
    assert client.post("/prefill", json={"prompt_tokens": tokens, "model_id": "m"}).status_code == 200
    body = client.post("/reconstruct", json={"tokens": tokens, "model_id": "m"}).json()
    assert body["coverage"] == 1.0
    assert [f["identity"]["token_span"] for f in body["fragments"]] == [[0, 127], [128, 255], [256, 299]]
    assert body["prefilled"] is False


def test_reconstruct_rejects_fragments_for_other_tokens() -> None:
    _server, client = make_server()
    client.post("/prefill", json={"prompt_tokens": prompt(256), "model_id": "m"})
    # Same first window, different second window: only the first may be reused,
    # even though a fragment sits at positions 128-255.
    other = prompt(128) + prompt(128, offset=10_000)
    body = client.post("/reconstruct", json={"tokens": other, "model_id": "m"}).json()
    assert [f["identity"]["token_span"] for f in body["fragments"]] == [[0, 127]]
    assert body["coverage"] == 0.5
    assert body["missing"] == [[128, 255]]
    # Another model's KV is never reused.
    assert client.post("/reconstruct", json={"tokens": prompt(256), "model_id": "x"}).json()["coverage"] == 0.0


def test_reconstruct_with_prefill_stores_absolute_spans() -> None:
    server, client = make_server()
    tokens = prompt(200)
    client.post("/prefill", json={"prompt_tokens": tokens[:128], "model_id": "m"})
    body = client.post("/reconstruct", json={"tokens": tokens, "model_id": "m", "prefill": True}).json()
    assert body["prefilled"] is True
    assert body["coverage"] == 1.0
    spans = sorted(tuple(f.identity.token_span) for f in server.node.fragments.values())
    assert spans == [(0, 127), (128, 199)]
    # The prefilled gap is now reusable without prefill.
    again = client.post("/reconstruct", json={"tokens": tokens, "model_id": "m"}).json()
    assert again["coverage"] == 1.0


def test_reconstruct_is_tenant_scoped() -> None:
    _server, client = make_server(authenticator=APIKeyAuthenticator(keyfile_text=KEYS))
    tokens = prompt(256)
    client.post("/prefill", json={"prompt_tokens": tokens, "model_id": "m"}, headers=auth("alice-key"))
    mine = client.post("/reconstruct", json={"tokens": tokens, "model_id": "m"}, headers=auth("alice-key")).json()
    theirs = client.post("/reconstruct", json={"tokens": tokens, "model_id": "m"}, headers=auth("bob-key")).json()
    assert mine["coverage"] == 1.0
    assert theirs["coverage"] == 0.0


def test_reconstruct_prefill_needs_write_scope() -> None:
    keys = "reader-key:reader:read\n"
    _server, client = make_server(authenticator=APIKeyAuthenticator(keyfile_text=keys))
    body = {"tokens": prompt(10), "model_id": "m", "prefill": True}
    assert client.post("/reconstruct", json=body, headers=auth("reader-key")).status_code == 403
    body["prefill"] = False
    assert client.post("/reconstruct", json=body, headers=auth("reader-key")).status_code == 200


def test_prefix_lookup_is_memoized_and_invalidated() -> None:
    server, client = make_server()
    client.post("/prefill", json={"prompt_tokens": prompt(256), "model_id": "m"})
    query = ",".join(map(str, prompt(300)))
    first = client.get(f"/prefix/lookup?model_id=m&tokens={query}").json()
    assert first == {
        "matched_tokens": 256,
        "total_tokens": 300,
        "full": False,
        "fragments": [token_hash(prompt(128)), token_hash(prompt(128, offset=128))],
        "cached": False,
    }
    second = client.post("/prefix/lookup", json={"tokens": prompt(300), "model_id": "m"}).json()
    assert second["cached"] is True and second["matched_tokens"] == 256
    with server.node.lock:
        server.node.remove_fragment(token_hash(prompt(128, offset=128)))
    third = client.post("/prefix/lookup", json={"tokens": prompt(300), "model_id": "m"}).json()
    assert third["cached"] is False and third["matched_tokens"] == 128


def test_prefix_lookup_rejects_bad_query() -> None:
    _server, client = make_server()
    assert client.get("/prefix/lookup?tokens=1,x,3").status_code == 400


def test_sessions_record_reads_per_tenant() -> None:
    _server, client = make_server(authenticator=APIKeyAuthenticator(keyfile_text=KEYS))
    stored = client.post("/prefill", json={"prompt_tokens": prompt(10), "model_id": "m"}, headers=auth("alice-key"))
    content_hash = stored.json()["fragments"][0]["identity"]["payload_hash"]
    headers = auth("alice-key") | {"X-Membrane-Session": "chat-1"}
    for _ in range(2):
        assert client.get(f"/retrieve?content_hash={content_hash}", headers=headers).json()["found"]
    session = client.get("/sessions/chat-1", headers=auth("alice-key")).json()
    assert session == {"session_id": "chat-1", "history": [content_hash, content_hash], "unique": 1}
    assert client.get("/sessions/chat-1", headers=auth("bob-key")).json()["history"] == []
    assert client.delete("/sessions/chat-1", headers=auth("bob-key")).json() == {"deleted": False}
    assert client.delete("/sessions/chat-1", headers=auth("alice-key")).json() == {"deleted": True}


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "prefix", "tokens": [1, 2, 3]},
        {"kind": "trace", "tool_name": "sql", "input": "select 1", "output": "1"},
        {
            "kind": "segment",
            "layer": 3,
            "head": 0,
            "token_span": [0, 15],
            "tensor_shape": [8, 16, 64],
            "data": base64.b64encode(b"kv" * 512).decode(),
        },
        {"kind": "artifact", "source_url": "https://example.com/doc", "data": base64.b64encode(b"text").decode()},
    ],
)
def test_typed_objects_round_trip(body: dict) -> None:
    _server, client = make_server()
    stored = client.post("/objects", json=body)
    assert stored.status_code == 200, stored.text
    content_hash = stored.json()["content_hash"]
    found = client.get(f"/objects/{content_hash}").json()
    assert found["kind"] == body["kind"]
    if body["kind"] == "prefix":
        assert found["object"]["tokens"] == [1, 2, 3]
    if body["kind"] == "trace":
        assert found["object"]["output"] == "1"
        assert found["object"]["tool_name"] == "sql"
    if "data" in body:
        assert found["data"] == body["data"]


def test_typed_object_errors() -> None:
    _server, client = make_server()
    assert client.post("/objects", json={"kind": "nope"}).status_code == 400
    assert client.post("/objects", json={"kind": "segment", "data": "!!"}).status_code == 400
    assert client.post("/objects", json=[1]).status_code == 400
    assert client.get("/objects/" + "0" * 64).status_code == 404


def stored_fragment(content_hash: str, span: tuple[int, int] = (0, 3), compat: str = "") -> Fragment:
    identity = PayloadIdentity(
        payload_hash=content_hash,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=span,
        dtype="float16",
        shape=(1, 1, 4, 1, 64),
    )
    return Fragment(
        identity=identity,
        payload_ref=None,
        payload_size=0,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
        fingerprint_compat=compat,
    )


def test_require_compat_refuses_other_models_and_stamps_prefill() -> None:
    server, client = make_server(services=ServiceOptions(require_compat="m:float16"))
    live = server.services.memory.validator.hash
    refused = client.post("/store", json={"fragment": to_dict(stored_fragment("a" * 32)), "is_primary": True})
    assert refused.status_code == 409
    accepted = client.post(
        "/store", json={"fragment": to_dict(stored_fragment("b" * 32, compat=live)), "is_primary": True}
    )
    assert accepted.status_code == 200
    prefilled = client.post("/prefill", json={"prompt_tokens": prompt(5), "model_id": "m"}).json()
    assert {f["fingerprint_compat"] for f in prefilled["fragments"]} == {live}


def test_store_refuses_identity_alias() -> None:
    _server, client = make_server()
    first = {"fragment": to_dict(stored_fragment("c" * 32, span=(0, 3))), "is_primary": True}
    alias = {"fragment": to_dict(stored_fragment("c" * 32, span=(128, 131))), "is_primary": True}
    assert client.post("/store", json=first).status_code == 200
    assert client.post("/store", json=first).status_code == 200  # idempotent
    assert client.post("/store", json=alias).status_code == 409


def test_heartbeat_advertises_role_and_gpu_load() -> None:
    _server, client = make_server(services=ServiceOptions(dynamic_roles=True))
    body = client.get("/heartbeat").json()
    assert body["role"] == "memory_host"
    assert body["gpu_load"] == 0.0


def test_client_methods() -> None:
    from membrane.client import MembraneClient

    _server, test_client = make_server()
    client = MembraneClient("http://testserver", transport=test_client)
    client.prefill(prompt(128), model_id="m")
    assert client.reconstruct(prompt(128), model_id="m", session_id="s")["coverage"] == 1.0
    assert client.session("s")["unique"] == 1
    assert client.prefix_lookup(prompt(200), model_id="m")["matched_tokens"] == 128
    assert client.route(tokens=prompt(200), model_id="m")["matched_tokens"] == 128
    stored = client.put_object({"kind": "prefix", "tokens": [7, 8]})
    assert client.get_object(stored["content_hash"])["object"]["tokens"] == [7, 8]
    assert client.get_object("0" * 64) is None


@pytest.mark.anyio
async def test_async_client_methods() -> None:
    import httpx

    from membrane.client import AsyncMembraneClient

    server, _test_client = make_server()
    transport = httpx.ASGITransport(app=server.transport.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        client = AsyncMembraneClient("http://testserver", client=http)
        await client.prefill(prompt(128), model_id="m")
        assert (await client.reconstruct(prompt(128), model_id="m"))["coverage"] == 1.0
        assert (await client.prefix_lookup(prompt(128), model_id="m"))["full"] is True
        assert (await client.route(content_hash=token_hash(prompt(128))))["placement"]["fetch_from"] == "mem"
        stored = await client.put_object({"kind": "trace", "tool_name": "t", "output": "o"})
        assert (await client.get_object(stored["content_hash"]))["kind"] == "trace"
        assert (await client.session("none"))["history"] == []
