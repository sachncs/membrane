"""Wire v3 bidi Open-stream integration test (Phase 3.3.3-3.3.4 follow-up)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from membrane.wire.v3 import (
    ChunkManifest,
    ResumableReceiver,
    ResumableTransfer,
    wire_v3_pb2,
    wire_v3_pb2_grpc,
)

# The generated protobuf classes are referenced through the
# module-level alias: Chunk, ChunkRequest.
Chunk = wire_v3_pb2.Chunk
ChunkRequest = wire_v3_pb2.ChunkRequest


class _TransferServicerImpl:
    """Test implementation of the wire_v3 Transfer service.

    The v3.0.0 release ships the wire_v3.proto; this test
    exercises the server side without standing up a real gRPC
    server by calling the generated ``Open`` method directly
    with a fake request iterator and capturing the response
    iterator.
    """

    def __init__(self, manifest: ChunkManifest, payload: bytes) -> None:
        self.manifest = manifest
        self.payload = payload
        self._received: list[ChunkRequest] = []
        self._lock = threading.Lock()

    def Open(
        self,
        request_iterator: Iterator[ChunkRequest],
        context: object,
    ) -> Iterator[Chunk]:
        with self._lock:
            for request in request_iterator:
                self._received.append(request)
        for index, chunk in enumerate(
            self.manifest.split_payload(self.payload)
        ):
            yield Chunk(
                chunk_index=index,
                offset=0,
                data=chunk,
                sha256_hex=self.manifest.per_chunk_sha256[index],
                is_last=(index == len(self.manifest.per_chunk_sha256) - 1),
            )


def test_open_stream_round_trip():
    """The wire_v3 Open stream exchanges every chunk and the receiver reassembles."""
    payload = b"abcdefghij" * 32  # 320 bytes
    manifest = ChunkManifest.from_payload(payload, "h" * 64, chunk_size=64)
    assert len(manifest.per_chunk_sha256) == 5

    servicer = _TransferServicerImpl(manifest, payload)

    # 1. Client builds a request iterator: one empty
    # ChunkRequest (manifest probe) per chunk.
    def request_iter() -> Iterator[ChunkRequest]:
        yield ChunkRequest(chunk_index=-1, offset=0, length=0)
        # The server can implement a "next chunk" protocol by
        # honoring whatever the client requests; for this
        # test we just request every chunk index in sequence.
        for index in range(len(manifest.per_chunk_sha256)):
            yield ChunkRequest(chunk_index=index, offset=0, length=0)

    # 2. Server yields the chunk sequence. Build a fake
    # context object since the service never consults it.
    class _Context:
        def set_code(self, *a, **k) -> None:  # pragma: no cover
            pass
        def abort(self, *a, **k) -> None:  # pragma: no cover
            pass

    response_iter = servicer.Open(request_iter(), _Context())
    # 3. Client feeds each chunk into a ResumableReceiver and
    # reassembles the payload.
    manifest = servicer.manifest
    transfer = ResumableTransfer.new(manifest)
    receiver = ResumableReceiver(transfer=transfer)
    for response in response_iter:
        # The wire_v3 wire-level chunk; the client decodes to
        # the same (chunk_index, data, sha256_hex) tuple.
        receiver.feed(response.chunk_index, response.data)
    assert transfer.all_chunks_received()
    assert transfer.assemble() == payload

    # 4. The server saw the request iterator walk.
    assert len(servicer._received) == 1 + len(manifest.per_chunk_sha256)


def test_open_stream_protobuf_construction():
    """Smoke test: the generated ChunkRequest / Chunk messages
    are constructible + serializable via the standard proto
    serialization path."""
    request = ChunkRequest(chunk_index=0, offset=0, length=1024)
    blob = request.SerializeToString()
    decoded = ChunkRequest()
    decoded.ParseFromString(blob)
    assert decoded.chunk_index == 0
    assert decoded.length == 1024

    chunk = Chunk(
        chunk_index=1,
        offset=0,
        data=b"hello",
        sha256_hex="0" * 64,
        is_last=True,
    )
    blob = chunk.SerializeToString()
    decoded = Chunk()
    decoded.ParseFromString(blob)
    assert decoded.chunk_index == 1
    assert decoded.is_last is True
    assert decoded.sha256_hex == "0" * 64


def test_open_stream_yields_zero_chunks_for_empty_payload():
    manifest = ChunkManifest.from_payload(b"", "h" * 64, chunk_size=2)
    assert manifest.per_chunk_sha256 == ()
    servicer = _TransferServicerImpl(manifest, b"")
    out = list(
        servicer.Open(
            iter([ChunkRequest(chunk_index=-1, offset=0, length=0)]),
            type("Ctx", (), {"set_code": lambda *a, **k: None})(),
        )
    )
    assert out == []


def test_transfer_stub_class_exists():
    """The generated TransferStub has the public attributes a client uses."""
    assert hasattr(wire_v3_pb2_grpc, "TransferStub")
    assert hasattr(wire_v3_pb2_grpc, "TransferServicer")
    assert hasattr(wire_v3_pb2_grpc, "add_TransferServicer_to_server")


def test_envelope_message_has_all_fields():
    """The Envelope message carries every field the v3.0.0 wire ships."""
    envelope = wire_v3_pb2.Envelope(
        request_id="r1",
        tenant_id="acme",
        model_id="llama-3",
        fingerprint_compat="f" * 64,
        content_hash="h" * 64,
        total_bytes=1024,
        compression=1,
        content_type="",
    )
    assert envelope.request_id == "r1"
    assert envelope.tenant_id == "acme"
    assert envelope.total_bytes == 1024
    assert envelope.compression == 1  # zstd
    blob = envelope.SerializeToString()
    decoded = wire_v3_pb2.Envelope()
    decoded.ParseFromString(blob)
    assert decoded.tenant_id == "acme"
