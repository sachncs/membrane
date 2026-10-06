"""The engine adapters and the gRPC surface against a live ``membrane serve`` process."""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from membrane.adapters.remote import HTTPClusterClient, HTTPSGLangClient, HTTPTrtClient, connect, prompt_handle
from membrane.adapters.sglang import SGLangKVEntry
from membrane.adapters.trtllm import TrtKVBlock
from membrane.adapters.vllm import LayerLoad, MembraneVLLMConnector
from membrane.runtime.concurrency import FREE_THREADED

ROOT = Path(__file__).resolve().parents[3]
#: The server child refuses --grpc-port on a free-threaded build (grpcio
#: would turn the GIL back on), so the gRPC check runs on GIL builds only.
GRPC = not FREE_THREADED or os.environ.get("PYTHON_GIL") == "1"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live() -> Iterator[tuple[str, int]]:
    """A ``membrane serve`` process with the REST and gRPC surfaces."""
    port, grpc_port = free_port(), free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "membrane",
            "serve",
            "--daemon",
            "--port",
            str(port),
            "--role",
            "both",
            *(["--grpc-port", str(grpc_port)] if GRPC else []),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1)
            break
        except OSError:
            if time.monotonic() > deadline or proc.poll() is not None:
                proc.kill()
                pytest.fail("membrane serve did not come up")
            time.sleep(0.2)
    try:
        yield f"http://127.0.0.1:{port}", grpc_port
    finally:
        proc.terminate()
        proc.wait(timeout=20)


class Request:
    def __init__(self, request_id: str, token_ids: tuple[int, ...]) -> None:
        self.request_id = request_id
        self.token_ids = token_ids
        self.model_id = "llama"


def test_vllm_connector_saves_and_reloads_through_membrane(live) -> None:
    url, _grpc = live
    layers, blocks, block_size, heads, dim = 2, 6, 4, 2, 8
    cache = [
        np.arange(2 * blocks * block_size * heads * dim, dtype=np.float16).reshape(2, blocks, block_size, heads, dim)
        + layer
        for layer in range(layers)
    ]
    tokens = tuple(range(8))

    writer = MembraneVLLMConnector(client=HTTPClusterClient(connect(url)), n_layers=layers, model_id="llama")
    request = Request("w1", tokens)
    assert writer.get_num_new_matched_tokens(request, None) == 0
    writer.update_state_after_alloc(request, (1, 4), None)
    for layer in range(layers):
        writer.save_kv(layer, cache, None)

    # Another engine instance finds the prompt and loads its blocks.
    reader_client = HTTPClusterClient(connect(url))
    reader = MembraneVLLMConnector(client=reader_client, n_layers=layers, model_id="llama")
    assert reader.get_num_new_matched_tokens(Request("r1", tokens), None) == len(tokens)
    handle = prompt_handle("llama", tokens)
    fetched = reader_client.fetch_layer(LayerLoad(1, handle), "llama", (1, 1, 1, 64), "float16")
    k = np.frombuffer(bytes(fetched.layers[0].k), dtype=np.float16)
    assert np.array_equal(k, cache[1][0:1, [1, 4]].ravel())
    assert reader.get_num_new_matched_tokens(Request("r2", (99,)), None) == 0


def test_sglang_and_trtllm_clients_round_trip(live) -> None:
    url, _grpc = live
    sglang = HTTPSGLangClient(connect(url))
    rows = (SGLangKVEntry(token_id=7, k=b"k7", v=b"v7"), SGLangKVEntry(token_id=8, k=b"k8", v=b"v8"))
    sglang.put("llama", "radix-node-1", rows)
    assert sglang.get("llama", "radix-node-1") == rows
    assert sglang.get("llama", "absent") == ()

    trt = HTTPTrtClient(connect(url))
    blocks = (TrtKVBlock(block_id=3, k=b"\x00" * 64, v=b"\x01" * 64),)
    trt.put("llama", "seq-9", blocks)
    assert trt.get("llama", "seq-9") == blocks


@pytest.mark.skipif(not GRPC, reason="gRPC is refused on a free-threaded build")
def test_grpc_prefill_and_decode(live) -> None:
    from membrane.disagg.grpc import (
        build_decode_request_message,
        build_prefill_request_message,
        decode_response_from_message,
        make_channel,
        make_stub,
        response_from_message,
    )
    from membrane.disagg.protocol import DecodeRequest, PrefillRequest

    _url, grpc_port = live
    stub = make_stub(make_channel(f"127.0.0.1:{grpc_port}"))
    prefill = PrefillRequest(request_id="p1", model_id="m", token_ids=tuple(range(40)))
    prefilled = response_from_message(stub.Prefill(build_prefill_request_message(prefill), timeout=10))
    assert prefilled.prompt_len == 40
    decode = DecodeRequest(request_id="d1", kv_handle=prefilled.kv_handle, model_id="m")
    decoded = decode_response_from_message(stub.Decode(build_decode_request_message(decode), timeout=10))
    assert decoded.finished is True
