"""Cluster clients that connect the engine adapters to a running Membrane node.

The vLLM, SGLang, and TensorRT-LLM adapters each talk to a small cluster
client. The in-memory clients keep KV in the engine's own process; the
HTTP clients here keep it in Membrane through the named KV bundle API
(``PUT``/``GET``/``HEAD /kv/{handle}``), so KV saved by one engine
instance is found by another and survives the engine:

* :class:`HTTPClusterClient` (vLLM): one bundle per prompt and layer,
  named ``<prompt handle>.<layer>``.
* :class:`HTTPSGLangClient`: token-indexed rows under the caller's handle.
* :class:`HTTPTrtClient`: KV blocks under the caller's handle.

Bundles are framed with :func:`pack_rows` (``MKV1``, then for each row its
index, K length and bytes, V length and bytes; little-endian).
"""

import logging
import struct
from collections.abc import Iterable
from typing import Any, override

from membrane.adapters import KVTensor, LayerKV
from membrane.adapters.sglang import SGLangClusterClient, SGLangKVEntry
from membrane.adapters.trtllm import TrtClusterClient, TrtKVBlock
from membrane.adapters.vllm import (
    LayerLoad,
    MatchedPrefix,
    MembraneClusterClient,
    placeholder_fingerprint,
    tensor_payload,
)
from membrane.client import MembraneClient
from membrane.prefix_cache import KVHandle

logger = logging.getLogger(__name__)

ROWS_MAGIC = b"MKV1"
ROW_HEADER = struct.Struct("<qI")


def pack_rows(rows: Iterable[tuple[int, bytes, bytes]]) -> bytes:
    """Frame ``(index, k, v)`` rows into one bundle.

    Args:
        rows: Rows in order.

    Returns:
        bytes: The framed bundle.
    """
    parts = [ROWS_MAGIC]
    for index, k, v in rows:
        parts += [ROW_HEADER.pack(index, len(k)), k, struct.pack("<I", len(v)), v]
    return b"".join(parts)


def unpack_rows(data: bytes) -> list[tuple[int, bytes, bytes]]:
    """Parse a bundle framed by :func:`pack_rows`.

    Args:
        data: The bundle.

    Returns:
        list[tuple[int, bytes, bytes]]: ``(index, k, v)`` rows in order.

    Raises:
        ValueError: On a malformed bundle.
    """
    if not data.startswith(ROWS_MAGIC):
        raise ValueError("not a Membrane KV bundle")
    rows = []
    offset = len(ROWS_MAGIC)
    try:
        while offset < len(data):
            index, k_len = ROW_HEADER.unpack_from(data, offset)
            offset += ROW_HEADER.size
            k = data[offset : offset + k_len]
            offset += k_len
            (v_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            v = data[offset : offset + v_len]
            offset += v_len
            if len(k) != k_len or len(v) != v_len:
                raise ValueError("truncated KV bundle")
            rows.append((index, k, v))
    except struct.error as exc:
        raise ValueError("truncated KV bundle") from exc
    return rows


def prompt_handle(model_id: str, token_ids: tuple[int, ...]) -> str:
    """The bundle name prefix for a prompt's KV.

    Args:
        model_id: Model identity.
        token_ids: The prompt.

    Returns:
        str: SHA-256 hex of ``(model_id, token_ids)``.
    """
    return KVHandle.for_tokens(model_id, token_ids).handle


class HTTPClusterClient(MembraneClusterClient):
    """vLLM cluster client backed by a Membrane node.

    A prompt's KV is found when its first layer is stored; each layer is
    its own bundle, so the runner loads layers as it needs them.

    Attributes:
        client: Membrane HTTP client.
    """

    def __init__(self, client: MembraneClient) -> None:
        """Wrap ``client``.

        Args:
            client: A :class:`~membrane.client.MembraneClient` for the node.
        """
        self.client = client

    @override
    def lookup_prefix(self, model_id: str, token_ids: tuple[int, ...]) -> MatchedPrefix:
        """Whether this prompt's KV is stored.

        Args:
            model_id: Model identity.
            token_ids: The prompt.

        Returns:
            MatchedPrefix: The whole prompt and its handle on a hit, else a miss.
        """
        if not token_ids:
            return MatchedPrefix(0, "")
        handle = prompt_handle(model_id, token_ids)
        if self.client.has_kv(f"{handle}.0", model_id=model_id):
            return MatchedPrefix(len(token_ids), handle)
        return MatchedPrefix(0, "")

    @override
    def start_load(self, kv_handle: str, layer_indices: tuple[int, ...]) -> tuple[LayerLoad, ...]:
        """Describe the layers to load (each is fetched on :meth:`fetch_layer`).

        Args:
            kv_handle: Handle from :meth:`lookup_prefix`.
            layer_indices: Layers the runner wants.

        Returns:
            tuple[LayerLoad, ...]: One load per layer.
        """
        return tuple(LayerLoad(layer_idx=i, kv_handle=kv_handle) for i in layer_indices)

    @override
    def fetch_layer(
        self, layer_load: LayerLoad, model_id: str, shape: tuple[int, int, int, int], dtype: str
    ) -> KVTensor:
        """Download one layer.

        Args:
            layer_load: The layer to fetch.
            model_id: Model identity.
            shape: Per-layer tensor shape.
            dtype: Element dtype.

        Returns:
            KVTensor: The layer (no layers when it is not stored).
        """
        data = self.client.get_kv(f"{layer_load.kv_handle}.{layer_load.layer_idx}", model_id=model_id)
        layers: tuple[LayerKV, ...] = ()
        if data is not None:
            index, k, v = unpack_rows(data)[0]
            layers = (LayerKV(layer_idx=index, k=memoryview(k), v=memoryview(v), head_range=(-1, -1), dtype=dtype),)
        return KVTensor(
            layers=layers,
            layer_range=(layer_load.layer_idx, layer_load.layer_idx),
            head_range=(-1, -1),
            token_span=(0, 0),
            shape=shape,
            fingerprint=placeholder_fingerprint(model_id, dtype),
        )

    @override
    def save_layer(
        self, layer: LayerKV, model_id: str, token_span: tuple[int, int], token_ids: tuple[int, ...] = ()
    ) -> None:
        """Upload one layer of a prompt's KV.

        Args:
            layer: The layer.
            model_id: Model identity.
            token_span: Token positions covered (informational).
            token_ids: The prompt the layer belongs to; without it the layer
                cannot be found again, so it is not stored.
        """
        if not token_ids:
            logger.debug("not saving layer %s: no prompt to file it under", layer.layer_idx)
            return
        handle = prompt_handle(model_id, token_ids)
        data = pack_rows([(layer.layer_idx, tensor_payload(layer.k), tensor_payload(layer.v))])
        self.client.put_kv(f"{handle}.{layer.layer_idx}", data, model_id=model_id)


class HTTPSGLangClient(SGLangClusterClient):
    """SGLang cluster client backed by a Membrane node.

    Attributes:
        client: Membrane HTTP client.
    """

    def __init__(self, client: MembraneClient) -> None:
        """Wrap ``client``.

        Args:
            client: A :class:`~membrane.client.MembraneClient` for the node.
        """
        self.client = client

    @override
    def get(self, model_id: str, handle: str) -> tuple[SGLangKVEntry, ...]:
        """Download the rows stored under ``handle``.

        Args:
            model_id: Model identity.
            handle: Bundle name.

        Returns:
            tuple[SGLangKVEntry, ...]: Rows in order (empty when absent).
        """
        data = self.client.get_kv(handle, model_id=model_id)
        if data is None:
            return ()
        return tuple(SGLangKVEntry(token_id=i, k=k, v=v) for i, k, v in unpack_rows(data))

    @override
    def put(self, model_id: str, handle: str, entries: tuple[SGLangKVEntry, ...]) -> None:
        """Upload rows under ``handle``.

        Args:
            model_id: Model identity.
            handle: Bundle name.
            entries: Rows in order.
        """
        self.client.put_kv(handle, pack_rows((e.token_id, e.k, e.v) for e in entries), model_id=model_id)


class HTTPTrtClient(TrtClusterClient):
    """TensorRT-LLM cluster client backed by a Membrane node.

    Attributes:
        client: Membrane HTTP client.
    """

    def __init__(self, client: MembraneClient) -> None:
        """Wrap ``client``.

        Args:
            client: A :class:`~membrane.client.MembraneClient` for the node.
        """
        self.client = client

    @override
    def get(self, model_id: str, handle: str) -> tuple[TrtKVBlock, ...]:
        """Download the blocks stored under ``handle``.

        Args:
            model_id: Model identity.
            handle: Bundle name.

        Returns:
            tuple[TrtKVBlock, ...]: Blocks in order (empty when absent).
        """
        data = self.client.get_kv(handle, model_id=model_id)
        if data is None:
            return ()
        return tuple(TrtKVBlock(block_id=i, k=k, v=v) for i, k, v in unpack_rows(data))

    @override
    def put(self, model_id: str, handle: str, blocks: tuple[TrtKVBlock, ...]) -> None:
        """Upload blocks under ``handle``.

        Args:
            model_id: Model identity.
            handle: Bundle name.
            blocks: Blocks in order.
        """
        self.client.put_kv(handle, pack_rows((b.block_id, b.k, b.v) for b in blocks), model_id=model_id)


def connect(base_url: str, api_key: str = "", **options: Any) -> MembraneClient:
    """Open a Membrane client for the HTTP cluster clients.

    Args:
        base_url: Node URL.
        api_key: Bearer key with ``write`` scope.
        **options: Further :class:`~membrane.client.MembraneClient` options.

    Returns:
        MembraneClient: The client.
    """
    return MembraneClient(base_url, api_key=api_key, **options)


__all__ = [
    "ROWS_MAGIC",
    "HTTPClusterClient",
    "HTTPSGLangClient",
    "HTTPTrtClient",
    "connect",
    "pack_rows",
    "prompt_handle",
    "unpack_rows",
]
