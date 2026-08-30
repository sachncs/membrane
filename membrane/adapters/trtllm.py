"""TensorRT-LLM KV transfer adapter (Phase 6).

Wires :class:`~membrane.adapters.KVAdapter` onto
TensorRT-LLM's :class:`KVCacheManager` and
:mod:`tensorrt_llm.runtime`. The adapter is optional: it only
materializes a real subclass of TRT-LLM's hooks when
TensorRT-LLM is importable. The tests run against a
duck-typed stub that mirrors the same surface.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from membrane.adapters import (
    BaseAdapter,
    KVTensor,
    LayerKV,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def _load_trtllm_base() -> type[Any] | None:
    """Import TRT-LLM's :class:`KVCacheManager` if installed.

    Returns:
        The class object when TRT-LLM is importable, otherwise
        ``None``.
    """
    try:
        from tensorrt_llm.runtime.kv_cache_manager import (  # type: ignore[import-not-found]
            KVCacheManager,
        )
    except ImportError:
        return None
    return KVCacheManager


_TRTLLM_BASE: type[Any] | None = _load_trtllm_base()


@dataclass(frozen=True)
class TrtKVBlock:
    """A single K/V block in TRT-LLM's block layout.

    Attributes:
        block_id: Block index the manager uses to address the
            block.
        k: K bytes (engine-raw).
        v: V bytes (engine-raw).
    """

    block_id: int
    k: bytes
    v: bytes


class TrtClusterClient:
    """TRT-LLM-facing cluster client."""

    def get(self, model_id: str, handle: str) -> tuple[TrtKVBlock, ...]:
        """Return the blocks for ``handle``.

        Args:
            model_id: Model identity.
            handle: Cluster-side handle.

        Returns:
            Tuple of :class:`TrtKVBlock` in order.
        """
        raise NotImplementedError

    def put(self, model_id: str, handle: str, blocks: tuple[TrtKVBlock, ...]) -> None:
        """Store ``blocks`` under ``handle``.

        Args:
            model_id: Model identity.
            handle: Cluster-side handle.
            blocks: K/V blocks in order.
        """
        raise NotImplementedError


class InMemoryTrtClient(TrtClusterClient):
    """In-memory TRT-LLM client used by tests + the v1 single-process path."""

    def __init__(self) -> None:
        self._by_handle: dict[tuple[str, str], tuple[TrtKVBlock, ...]] = {}
        self._lock = threading.RLock()

    def seed(self, model_id: str, handle: str, blocks: tuple[TrtKVBlock, ...]) -> None:
        """Pre-populate blocks for ``handle``.

        Args:
            model_id: Model identity.
            handle: Cluster-side handle.
            blocks: K/V blocks in order.
        """
        with self._lock:
            self._by_handle[(model_id, handle)] = blocks

    def get(self, model_id: str, handle: str) -> tuple[TrtKVBlock, ...]:
        with self._lock:
            return self._by_handle.get((model_id, handle), ())

    def put(self, model_id: str, handle: str, blocks: tuple[TrtKVBlock, ...]) -> None:
        with self._lock:
            self._by_handle[(model_id, handle)] = blocks


class MembraneTrtAdapter(BaseAdapter):
    """TensorRT-LLM-flavored :class:`KVAdapter` (Phase 6).

    TRT-LLM's :class:`KVCacheManager` addresses K/V by block
    id; the adapter translates between a contiguous
    ``token_span`` and the manager's block table.
    """

    BLOCK_SIZE: int = 64

    def __init__(self, kv_backend: Any) -> None:
        """Initialize the adapter.

        Args:
            kv_backend: A :class:`KVBackend` or ``None`` for
                the TRT-LLM-only path.
        """
        self.kv_backend = kv_backend

    def extract(
        self,
        model: Any,
        layer_range: tuple[int, int],
        head_range: tuple[int, int],
        token_span: tuple[int, int],
    ) -> KVTensor:
        """Read K/V blocks from a TRT-LLM :class:`KVCacheManager`.

        Args:
            model: A TRT-LLM model handle whose
                ``kv_cache_manager`` is a
                :class:`KVCacheManager`.
            layer_range: Inclusive ``(start, end)``.
            head_range: Inclusive ``(start, end)``.
            token_span: Inclusive ``(start, end)``.

        Returns:
            KVTensor: One :class:`LayerKV` per layer with the
            block bytes for ``token_span``.
        """
        manager = getattr(model, "kv_cache_manager", None)
        if manager is None:
            return KVTensor(
                layers=(),
                layer_range=layer_range,
                head_range=head_range,
                token_span=token_span,
                shape=(1, 1, 1, 64),
                fingerprint=_placeholder_fingerprint(),
            )
        block_indices = self._blocks_for(token_span)
        layers = self._read_blocks(manager, layer_range, block_indices)
        return KVTensor(
            layers=layers,
            layer_range=layer_range,
            head_range=head_range,
            token_span=token_span,
            shape=(layers[0].k.__sizeof__() if layers else 1, 1, 1, 64),
            fingerprint=_placeholder_fingerprint(),
        )

    def import_into(
        self,
        model: Any,
        tensor: KVTensor,
        layer_range: tuple[int, int],
    ) -> None:
        """Install a K/V bundle into a TRT-LLM manager.

        Args:
            model: A TRT-LLM model handle.
            tensor: Bundle produced by :func:`extract`.
            layer_range: Inclusive ``(start, end)``.
        """
        manager = getattr(model, "kv_cache_manager", None)
        if manager is None:
            logger.debug("MembraneTrtAdapter.import_into: no kv_cache_manager on model")
            return
        block_indices = self._blocks_for(tensor.token_span)
        self._write_blocks(manager, tensor.layers, block_indices)

    def validate(self, tensor: KVTensor) -> ValidationResult:
        """Default BaseAdapter validation.

        Args:
            tensor: Bundle to validate.

        Returns:
            ValidationResult: Outcome of the checks.
        """
        return super().validate(tensor)

    def _blocks_for(self, token_span: tuple[int, int]) -> tuple[int, ...]:
        """Translate a token span into a tuple of block indices.

        Args:
            token_span: Inclusive ``(start, end)``.

        Returns:
            Tuple of block indices covering the span.
        """
        start, end = token_span
        if end < start:
            return ()
        return tuple(range(start // self.BLOCK_SIZE, end // self.BLOCK_SIZE + 1))

    @staticmethod
    def _read_blocks(
        manager: Any,
        layer_range: tuple[int, int],
        block_indices: tuple[int, ...],
    ) -> tuple[LayerKV, ...]:
        """Read per-layer K/V bytes for the given blocks.

        Args:
            manager: TRT-LLM KVCacheManager.
            layer_range: Inclusive ``(start, end)``.
            block_indices: Tuple of block indices.

        Returns:
            Tuple of :class:`LayerKV` rows.
        """
        layers: list[LayerKV] = []
        for layer in range(layer_range[0], layer_range[1] + 1):
            k_bytes = _read_block(manager, layer, "k", block_indices)
            v_bytes = _read_block(manager, layer, "v", block_indices)
            layers.append(
                LayerKV(
                    layer_idx=layer,
                    k=k_bytes,
                    v=v_bytes,
                    head_range=(-1, -1),
                    dtype="float16",
                )
            )
        return tuple(layers)

    @staticmethod
    def _write_blocks(
        manager: Any,
        layers: tuple[LayerKV, ...],
        block_indices: tuple[int, ...],
    ) -> None:
        """Write the K/V blocks back into ``manager``.

        Args:
            manager: TRT-LLM KVCacheManager.
            layers: Layers to write.
            block_indices: Tuple of block indices.
        """
        for layer in layers:
            _write_block(manager, layer.layer_idx, "k", layer.k, block_indices)
            _write_block(manager, layer.layer_idx, "v", layer.v, block_indices)


def _read_block(manager: Any, layer: int, kind: str, block_indices: tuple[int, ...]) -> bytes:
    """Read a K or V block from ``manager``.

    Args:
        manager: TRT-LLM KVCacheManager.
        layer: Layer index.
        kind: ``"k"`` or ``"v"``.
        block_indices: Block indices to read.

    Returns:
        bytes: Concatenated block bytes.
    """
    if hasattr(manager, "get_block_bytes"):
        return bytes(
            b"".join(manager.get_block_bytes(layer=layer, kind=kind, block_id=b) for b in block_indices)
        )
    if hasattr(manager, "k_cache") and hasattr(manager, "v_cache"):
        buf = manager.k_cache if kind == "k" else manager.v_cache
        if isinstance(buf, dict):
            return b"".join(bytes(buf.get(layer, {}).get(b, b"")) for b in block_indices)
    return b""


def _write_block(manager: Any, layer: int, kind: str, payload: Any, block_indices: tuple[int, ...]) -> None:
    """Write a K or V block into ``manager``.

    Args:
        manager: TRT-LLM KVCacheManager.
        layer: Layer index.
        kind: ``"k"`` or ``"v"``.
        payload: Bytes to write.
        block_indices: Block indices to write.
    """
    if hasattr(manager, "set_block_bytes"):
        manager.set_block_bytes(layer=layer, kind=kind, payload=bytes(payload), block_ids=block_indices)
        return
    if hasattr(manager, "k_cache") and hasattr(manager, "v_cache"):
        buf = manager.k_cache if kind == "k" else manager.v_cache
        if isinstance(buf, dict):
            data = bytes(payload)
            per_block = max(len(data) // max(len(block_indices), 1), 1)
            for i, block_id in enumerate(block_indices):
                chunk = data[i * per_block : (i + 1) * per_block]
                buf.setdefault(layer, {})[block_id] = chunk


def _placeholder_fingerprint() -> Any:
    """Return a placeholder fingerprint for TRT-LLM bundles.

    Returns:
        ModelCompatibilityFingerprint: Default.
    """
    from membrane.compat import compat_hash

    return compat_hash(model_id="trtllm", dtype="float16")


__all__ = [
    "TRTLLM_AVAILABLE",
    "InMemoryTrtClient",
    "MembraneTrtAdapter",
    "TrtClusterClient",
    "TrtKVBlock",
]


TRTLLM_AVAILABLE: bool = _TRTLLM_BASE is not None
