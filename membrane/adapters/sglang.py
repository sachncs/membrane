"""SGLang KV transfer adapter (Phase 6).

Wires :class:`~membrane.adapters.KVAdapter` onto SGLang's
:mod:`sglang.srt.mem_cache.radix_cache` plus the
:mod:`sglang.srt.mem_cache.memory_pool` token-to-KV pool. The
adapter is optional: it only materializes a real subclass of
SGLang's hooks when SGLang is importable. The tests run
against a duck-typed stub that mirrors the same surface.
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


def _load_sglang_pool_base() -> type[Any] | None:
    """Import SGLang's :class:`TokenToKVPool` if installed.

    Returns:
        The class object when SGLang is importable, otherwise
        ``None``.
    """
    try:
        from sglang.srt.mem_cache.memory_pool import (  # type: ignore[import-not-found]
            TokenToKVPool,
        )
    except ImportError:
        return None
    return TokenToKVPool


_SGLANG_POOL_BASE: type[Any] | None = _load_sglang_pool_base()


@dataclass(frozen=True)
class SGLangKVEntry:
    """A single K/V row indexed by token position.

    Attributes:
        token_id: Token id for this row.
        k: K bytes (engine-raw).
        v: V bytes (engine-raw).
    """

    token_id: int
    k: bytes
    v: bytes


class SGLangClusterClient:
    """SGLang-facing cluster client.

    SGLang's runtime operates on token-indexed K/V rows; the
    client stores them by ``(model_id, token_span_start)``
    handle and returns the rows in order.
    """

    def get(self, model_id: str, handle: str) -> tuple[SGLangKVEntry, ...]:
        """Return the rows for ``handle``.

        Args:
            model_id: Model identity.
            handle: Cluster-side handle.

        Returns:
            Tuple of K/V rows in order.
        """
        raise NotImplementedError

    def put(self, model_id: str, handle: str, entries: tuple[SGLangKVEntry, ...]) -> None:
        """Store ``entries`` under ``handle``.

        Args:
            model_id: Model identity.
            handle: Cluster-side handle.
            entries: Token-indexed rows in order.
        """
        raise NotImplementedError


class InMemorySGLangClient(SGLangClusterClient):
    """In-memory SGLang client used by tests + the v1 single-process path."""

    def __init__(self) -> None:
        self._by_handle: dict[tuple[str, str], tuple[SGLangKVEntry, ...]] = {}
        self._lock = threading.RLock()

    def seed(self, model_id: str, handle: str, entries: tuple[SGLangKVEntry, ...]) -> None:
        """Pre-populate rows for ``handle``.

        Args:
            model_id: Model identity.
            handle: Cluster-side handle.
            entries: Token-indexed rows in order.
        """
        with self._lock:
            self._by_handle[(model_id, handle)] = entries

    def get(self, model_id: str, handle: str) -> tuple[SGLangKVEntry, ...]:
        with self._lock:
            return self._by_handle.get((model_id, handle), ())

    def put(self, model_id: str, handle: str, entries: tuple[SGLangKVEntry, ...]) -> None:
        with self._lock:
            self._by_handle[(model_id, handle)] = entries


class MembraneSGLangAdapter(BaseAdapter):
    """SGLang-flavored :class:`KVAdapter` (Phase 6).

    SGLang's radix-cache stores K/V rows in a
    :class:`TokenToKVPool` indexed by token position. The
    adapter materializes :class:`LayerKV` rows by slicing the
    pool along the token axis for the requested ``token_span``.
    """

    def __init__(self, kv_backend: Any) -> None:
        """Initialize the adapter.

        Args:
            kv_backend: A :class:`KVBackend` or ``None`` for
                the SGLang-only path.
        """
        self.kv_backend = kv_backend

    def extract(
        self,
        model: Any,
        layer_range: tuple[int, int],
        head_range: tuple[int, int],
        token_span: tuple[int, int],
    ) -> KVTensor:
        """Read K/V rows from an SGLang :class:`TokenToKVPool`.

        Args:
            model: An SGLang model handle whose ``kv_pool`` is
                a :class:`TokenToKVPool`.
            layer_range: Inclusive ``(start, end)``.
            head_range: Inclusive ``(start, end)``.
            token_span: Inclusive ``(start, end)``.

        Returns:
            KVTensor: One :class:`LayerKV` per layer with the
            rows that cover ``token_span``.
        """
        pool = getattr(model, "kv_pool", None)
        if pool is None:
            return KVTensor(
                layers=(),
                layer_range=layer_range,
                head_range=head_range,
                token_span=token_span,
                shape=(1, 1, 1, 64),
                fingerprint=_placeholder_fingerprint(),
            )
        rows = self._read_rows(pool, layer_range, token_span)
        return KVTensor(
            layers=rows,
            layer_range=layer_range,
            head_range=head_range,
            token_span=token_span,
            shape=(rows[0].k.__sizeof__() if rows else 1, 1, 1, 64),
            fingerprint=_placeholder_fingerprint(),
        )

    def import_into(
        self,
        model: Any,
        tensor: KVTensor,
        layer_range: tuple[int, int],
    ) -> None:
        """Install a K/V bundle into an SGLang pool.

        Args:
            model: An SGLang model handle whose ``kv_pool`` is
                a :class:`TokenToKVPool`.
            tensor: Bundle produced by :func:`extract`.
            layer_range: Inclusive ``(start, end)``.
        """
        pool = getattr(model, "kv_pool", None)
        if pool is None:
            logger.debug("MembraneSGLangAdapter.import_into: no kv_pool on model")
            return
        self._write_rows(pool, tensor.layers, layer_range)

    def validate(self, tensor: KVTensor) -> ValidationResult:
        """Default BaseAdapter validation.

        Args:
            tensor: Bundle to validate.

        Returns:
            ValidationResult: Outcome of the checks.
        """
        return super().validate(tensor)

    @staticmethod
    def _read_rows(
        pool: Any,
        layer_range: tuple[int, int],
        token_span: tuple[int, int],
    ) -> tuple[LayerKV, ...]:
        """Slice ``pool`` for ``token_span`` per layer.

        Args:
            pool: Engine-resident K/V pool.
            layer_range: Inclusive ``(start, end)``.
            token_span: Inclusive ``(start, end)``.

        Returns:
            Tuple of :class:`LayerKV` rows.
        """
        rows: list[LayerKV] = []
        for layer in range(layer_range[0], layer_range[1] + 1):
            k_bytes = _slice_pool(pool, layer, "k", token_span)
            v_bytes = _slice_pool(pool, layer, "v", token_span)
            rows.append(
                LayerKV(
                    layer_idx=layer,
                    k=k_bytes,
                    v=v_bytes,
                    head_range=(-1, -1),
                    dtype="float16",
                )
            )
        return tuple(rows)

    @staticmethod
    def _write_rows(pool: Any, layers: tuple[LayerKV, ...], layer_range: tuple[int, int]) -> None:
        """Write the K/V rows back into ``pool``.

        Args:
            pool: Engine-resident K/V pool.
            layers: Layers to write.
            layer_range: Inclusive ``(start, end)``.
        """
        for layer in layers:
            _store_pool(pool, layer.layer_idx, "k", layer.k)
            _store_pool(pool, layer.layer_idx, "v", layer.v)
        del layer_range


def _slice_pool(pool: Any, layer: int, kind: str, token_span: tuple[int, int]) -> bytes:
    """Slice a pool's K or V bytes for the given layer.

    Args:
        pool: The pool to read from.
        layer: Layer index.
        kind: ``"k"`` or ``"v"``.
        token_span: Inclusive ``(start, end)``.

    Returns:
        bytes: The row bytes.
    """
    start, end = token_span
    if hasattr(pool, "get_kv_bytes"):
        return bytes(pool.get_kv_bytes(layer=layer, kind=kind, start=start, end=end))
    if hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        buf = pool.k_buffer if kind == "k" else pool.v_buffer
        return b"".join(bytes(buf.get(token, b"")) for token in range(start, end + 1))
    return b""


def _store_pool(pool: Any, layer: int, kind: str, payload: Any) -> None:
    """Write ``payload`` into ``pool`` for the given layer.

    Args:
        pool: The pool to write to.
        layer: Layer index.
        kind: ``"k"`` or ``"v"``.
        payload: Bytes to write.
    """
    if hasattr(pool, "set_kv_bytes"):
        pool.set_kv_bytes(layer=layer, kind=kind, payload=bytes(payload))
        return
    if hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        buf = pool.k_buffer if kind == "k" else pool.v_buffer
        if isinstance(buf, dict):
            data = bytes(payload)
            for i, token in enumerate(sorted(buf.keys())):
                if i * (len(data) // max(len(buf), 1)) < len(data):
                    chunk = data[i : i + 1]
                    buf[token] = chunk


def _placeholder_fingerprint() -> Any:
    """Return a placeholder fingerprint for SGLang bundles.

    Returns:
        ModelCompatibilityFingerprint: Default.
    """
    from membrane.compat import compat_hash

    return compat_hash(model_id="sglang", dtype="float16")


__all__ = [
    "SGLANG_AVAILABLE",
    "InMemorySGLangClient",
    "MembraneSGLangAdapter",
    "SGLangClusterClient",
    "SGLangKVEntry",
]


SGLANG_AVAILABLE: bool = _SGLANG_POOL_BASE is not None
