"""``/delete``, ``/tombstone``, ``/purge``, and ``/verify`` over HTTP (admin scope)."""

import hashlib
import time

from fastapi.testclient import TestClient

from membrane.auth.apikey import APIKeyAuthenticator
from membrane.node import Node
from membrane.server import Server
from tests.conftest import make_fragment

KEYS = "root-key:root:admin\nuser-key:user:read,write\n"
ROOT = {"Authorization": "Bearer root-key"}
H = "d" * 32


def server_with_fragment() -> tuple[Server, TestClient]:
    server = Server(node=Node("del"), port=0, load_hooks=False, authenticator=APIKeyAuthenticator(keyfile_text=KEYS))
    server.node.content_store.put(f"blob-{H}", b"kv-bytes")
    assert server.node.store(make_fragment(H, (0, 3), size=8))
    return server, TestClient(server.transport.app)


def test_delete_removes_and_tombstones() -> None:
    server, client = server_with_fragment()
    body = {"content_hash": H, "node_id": "del", "tombstone_until": time.time() + 60}
    assert client.post("/delete", json=body, headers={"Authorization": "Bearer user-key"}).status_code == 403
    assert client.post("/delete", json=body, headers=ROOT).json() == {"success": True, "content_hash": H}
    assert H not in server.node.fragments
    assert server.tombstones.is_active(H)
    assert client.post("/delete", json=body, headers=ROOT).json()["noop"] is True


def test_tombstone_then_purge() -> None:
    server, client = server_with_fragment()
    marked = client.post(
        "/tombstone", json={"content_hash": H, "until": time.time() + 60, "node_id": "del"}, headers=ROOT
    )
    assert marked.json() == {"success": True, "content_hash": H}
    assert H in server.node.fragments  # a tombstone alone does not remove
    assert client.post("/purge", json={"content_hash": H}, headers=ROOT).json()["success"] is True
    assert H not in server.node.fragments
    assert client.post("/purge", json={"content_hash": H}, headers=ROOT).json()["success"] is False


def test_verify_hashes_the_fragments_own_blob() -> None:
    _server, client = server_with_fragment()
    good = hashlib.sha256(b"kv-bytes").hexdigest()
    body = client.post(
        "/verify", json={"content_hash": H, "claimed_size": 8, "claimed_sha256_hex": good}, headers=ROOT
    ).json()
    assert body["success"] is True and body["bytes_match"] is True and body["sha256"] == good
    wrong = client.post(
        "/verify", json={"content_hash": H, "claimed_size": 8, "claimed_sha256_hex": "0" * 64}, headers=ROOT
    )
    assert wrong.json()["bytes_match"] is False
    size = client.post("/verify", json={"content_hash": H, "claimed_size": 9, "claimed_sha256_hex": good}, headers=ROOT)
    assert size.json()["reason"] == "size mismatch"
    missing = client.post(
        "/verify", json={"content_hash": "x" * 32, "claimed_size": 1, "claimed_sha256_hex": good}, headers=ROOT
    )
    assert missing.json()["reason"] == "fragment missing"
