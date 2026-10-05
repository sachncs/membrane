"""Large blobs upload in verified chunks and resume after an interruption."""

import hashlib
import json
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from membrane.errors import NetworkError
from membrane.network.peer import Peer, RawResponse
from membrane.node import Node
from membrane.server import Server
from membrane.wire.v3.chunks import ChunkManifest


class AppTransport:
    """Peer transport that calls an in-process app; can fail chosen requests."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.paths: list[str] = []
        self.fail_paths: set[str] = set()

    def request(self, method, url, body, headers, timeout_sec):
        path = urlsplit(url).path + (f"?{urlsplit(url).query}" if urlsplit(url).query else "")
        self.paths.append(f"{method} {path}")
        if path in self.fail_paths:
            self.fail_paths.discard(path)
            raise NetworkError("connection dropped")
        response = self.client.request(method, path, content=body, headers=headers)
        if response.status_code >= 400:
            raise NetworkError(f"HTTP {response.status_code}")
        return json.loads(response.text) if response.text else {}

    def request_bytes(self, method, url, body, headers, timeout_sec):
        response = self.client.request(method, urlsplit(url).path, headers=headers)
        return RawResponse(response.status_code, {k.lower(): v for k, v in response.headers.items()}, response.content)


def test_interrupted_upload_resumes_with_only_missing_chunks() -> None:
    server = Server(node=Node("up"), port=0, load_hooks=False)
    transport = AppTransport(TestClient(server.transport.app))
    peer = Peer("http://peer:8080", transport=transport, max_retries=1, retry_delay_sec=0.0)
    data = bytes(range(256)) * (10 * 4096)  # 10 MiB: resumable path
    transport.fail_paths = {"/blobs/big-1/upload/1"}
    assert peer.put_blob("big-1", data) is False  # interrupted at chunk 1
    sent_first = [p for p in transport.paths if "/upload/" in p]
    transport.paths.clear()
    assert peer.put_blob("big-1", data) is True
    resumed = [p for p in transport.paths if "/upload/" in p]
    assert "PUT /blobs/big-1/upload/0" in sent_first and "PUT /blobs/big-1/upload/0" not in resumed
    assert resumed[0] == "PUT /blobs/big-1/upload/1"
    assert server.node.content_store.get("big-1") == data


def test_bad_chunks_and_digests_are_rejected() -> None:
    client = TestClient(Server(node=Node("up2"), port=0, load_hooks=False).transport.app)
    data = b"x" * 3000
    manifest = ChunkManifest.from_payload(data, content_hash="b2", chunk_size=1000)
    begin = {"chunk_size": 1000, "total_bytes": 3000, "chunks": list(manifest.per_chunk_sha256), "sha256": "0" * 64}
    assert client.post("/blobs/blob-b2/upload", json=begin).json() == {"stored": False, "received": []}
    assert client.put("/blobs/blob-b2/upload/0", content=b"y" * 1000).status_code == 400  # wrong chunk
    for i in range(2):
        assert client.put(f"/blobs/blob-b2/upload/{i}", content=data[i * 1000 : (i + 1) * 1000]).status_code == 200
    assert client.post("/blobs/blob-b2/upload", json=begin).json()["received"] == [0, 1]  # resumable
    last = client.put("/blobs/blob-b2/upload/2", content=data[2000:])
    assert last.status_code == 400 and "digest" in last.json()["error"]  # whole-payload digest wrong
    good = {**begin, "sha256": hashlib.sha256(data).hexdigest()}
    client.post("/blobs/blob-b3/upload", json=good)
    for i in range(3):
        response = client.put(f"/blobs/blob-b3/upload/{i}", content=data[i * 1000 : (i + 1) * 1000])
    assert response.json() == {"stored": True, "received": []}
    assert client.get("/blobs/blob-b3").content == data
    assert client.post("/blobs/blob-b3/upload", json=good).json()["stored"] is True  # already here
