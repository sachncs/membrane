"""/sync copies bytes through the authenticated peer client; /inventory pages."""

import json
import socket
import time

from fastapi.testclient import TestClient

from membrane.auth.apikey import APIKeyAuthenticator, generate_key
from membrane.network.peer import PeerCredentials, set_default_peer_credentials
from membrane.node import Node
from membrane.security.url_allowlist import configure as configure_allowlist
from membrane.security.url_allowlist import reset_default_allowlist
from membrane.server import Server
from membrane.sync import DeltaSync
from tests.conftest import make_fragment


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_inventory_pages() -> None:
    node = Node("pg")
    for i in range(25):
        node.store(make_fragment(f"h{i:03d}"))
    client = TestClient(Server(node=node, port=0, load_hooks=False).transport.app)
    seen, cursor = [], ""
    while True:
        page = client.get("/inventory", params={"limit": 10, "after": cursor}).json()
        seen.extend(page["digest"])
        cursor = page["next"]
        if not cursor:
            break
    assert seen == sorted(f"h{i:03d}" for i in range(25))
    assert len(client.get("/inventory").json()["digest"]) == 25  # unpaged default


def test_delta_plan_from_digests() -> None:
    plan = DeltaSync.plan_from_digests("src", "dst", {"a": 2, "b": 1, "c": 1}, {"a": 1, "c": 1})
    assert plan.missing_hashes == ["b"] and plan.outdated_hashes == ["a"]


def test_sync_pulls_bytes_from_an_authenticated_source() -> None:
    peer_key, peer_line = generate_key("peers", ["admin"])
    source_node = Node("src")
    for i in range(3):
        frag = make_fragment(f"s{i}")
        source_node.content_store.put(frag.payload_ref, f"kv-{i}".encode() * 400)
        source_node.store(frag)
    port = free_port()
    source = Server(
        node=source_node,
        host="127.0.0.1",
        port=port,
        authenticator=APIKeyAuthenticator(peer_line + "\n"),
        load_hooks=False,
    )
    source.start()
    try:
        set_default_peer_credentials(PeerCredentials(bearer_token=peer_key))
        configure_allowlist(allowlist=["127.0.0.1"], allowed_networks=["127.0.0.0/8"])
        target = Server(node=Node("dst"), port=0, load_hooks=False)
        client = TestClient(target.transport.app)
        deadline = time.monotonic() + 10
        while True:
            body = client.post("/sync", json={"source_url": f"http://127.0.0.1:{port}"}).json()
            if body.get("success") or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        assert body["success"] and sorted(body["transferred"]) == ["s0", "s1", "s2"], json.dumps(body)
        for i in range(3):
            found = client.get("/retrieve", params={"content_hash": f"s{i}"}).json()
            assert found["found"] is True  # bytes arrived, not just metadata
            assert target.node.content_store.get(f"blob-s{i}") == f"kv-{i}".encode() * 400
    finally:
        source.stop(2.0)
        set_default_peer_credentials(PeerCredentials())
        reset_default_allowlist()
