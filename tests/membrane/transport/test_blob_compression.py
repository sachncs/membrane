"""Blob transfers compress on the wire; decompression is bounded."""

import hashlib

import pytest
from fastapi.testclient import TestClient

from membrane.codec import CompressionTransport, method_available
from membrane.node import Node
from membrane.runtime.settings import ServerSettings, SettingsError
from membrane.server import Server
from membrane.transport.ops import MAX_BODY_BYTES


@pytest.fixture
def client() -> TestClient:
    return TestClient(Server(node=Node("c0"), port=0, load_hooks=False).transport.app)


def test_compressed_upload_is_stored_uncompressed(client: TestClient) -> None:
    data = b"kv-" * 10_000
    body = CompressionTransport("zstd").compress(data)
    headers = {"X-Content-SHA256": hashlib.sha256(data).hexdigest(), "X-Membrane-Compression": "zstd"}
    assert client.put("/blobs/blob-z1", content=body, headers=headers).status_code == 200
    raw = client.get("/blobs/blob-z1")
    assert raw.content == data and "x-membrane-compression" not in raw.headers


def test_download_compresses_when_asked(client: TestClient) -> None:
    data = b"kv-" * 10_000
    client.put("/blobs/blob-z2", content=data, headers={"X-Content-SHA256": hashlib.sha256(data).hexdigest()})
    response = client.get("/blobs/blob-z2", headers={"X-Membrane-Accept-Compression": "zstd"})
    assert response.headers["x-membrane-compression"] == "zstd"
    assert len(response.content) < len(data)
    assert CompressionTransport().decompress(response.content) == data
    assert response.headers["x-content-sha256"] == hashlib.sha256(data).hexdigest()


def test_decompression_bomb_is_rejected(client: TestClient) -> None:
    bomb = CompressionTransport("zstd").compress(b"\0" * (MAX_BODY_BYTES + 1))
    assert len(bomb) < 100_000
    headers = {"X-Content-SHA256": "0" * 64, "X-Membrane-Compression": "zstd"}
    response = client.put("/blobs/blob-bomb", content=bomb, headers=headers)
    assert response.status_code == 400 and "exceeds" in response.json()["detail"]


def test_transfer_compression_setting() -> None:
    assert method_available("zstd") and method_available("raw")
    with pytest.raises(SettingsError, match="transfer compression"):
        ServerSettings(transfer_compression="brotli")
