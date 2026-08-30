"""LMCache-backed :class:`ContentStore` (Phase 0.2).

The v2.0+ canonical byte substrate is `LMCache
<https://github.com/LMCache/LMCache>`_. This module wraps
:class:`lmcache.v1.storage_backend.local_cpu_backend.LocalCPUBackend`
as a concrete :class:`membrane.content_store.ContentStore` so the
existing ``put``/``get``/``delete``/``size`` surface keeps working
while the underlying physical layer is LMCache's per-key allocator.

LMCache's high-level
:class:`~lmcache.v1.cache_engine.LMCacheEngine` is GPU-coupled
(it carries tensors from HBM to the storage backends) and
expects a ``gpu_connector`` for ``store()``. Operators that want
the full engine path go through the
:mod:`membrane.engines` adapters in Phase 5+. The
``ContentStore`` layer uses LMCache's lower-level
``LocalCPUBackend`` directly, which gives us the same allocator
+ chunked-blob mechanics without the GPU-stack dependency.

The module imports :mod:`lmcache` lazily so a Membrane install
without the ``[lmcache]`` extras still works.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _hex_key_to_int(key: str) -> int:
    """Map an arbitrary key to a stable, signed 64-bit int.

    LMCache's ``LocalCPUBackend`` is keyed on a 64-bit integer;
    the first 8 bytes of ``sha256(key)`` are interpreted as a
    big-endian unsigned value and sign-folded into a signed int so
    Python's int hash scheme does not perturb the value.

    Args:
        key: Opaque key string.

    Returns:
        int: 64-bit signed int.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    if raw >= 1 << 63:
        raw -= 1 << 64
    return raw


class LMCacheContentStore:
    """``ContentStore`` wrapper around LMCache's :class:`LocalCPUBackend`.

    Optional LMCache install: this class fails fast with a clear
    ``ImportError`` when LMCache is missing. Use
    :class:`membrane.content_store.InProcessBytes` or
    :class:`membrane.content_store.FilesystemBlob` for the default
    storage path; LMCache is the production v2.0+ deployment story.

    Attributes:
        config: Reserved for future LMCache engine config
            overrides; the v1 of this store keeps the
            signature stable so callers can pass an empty dict.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the LMCache-backed content store.

        Args:
            config: Optional LMCache engine config overrides;
                reserved for Phase 5+ engine integrations.

        Raises:
            ImportError: When LMCache is not installed.
            RuntimeError: When the LMCache backend fails to
                initialise.
        """
        try:
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.metadata import LMCacheMetadata
            from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
        except ImportError as exc:
            raise ImportError(
                "LMCache is required for LMCacheContentStore; install "
                "with `pip install membrane[lmcache]`."
            ) from exc
        from lmcache.v1.memory_management import TensorMemoryObj  # noqa: E402  -- used in _make_object

        self.config = config or {}
        self._lock = threading.RLock()
        try:
            engine_config = LMCacheEngineConfig.from_defaults()
            metadata = LMCacheMetadata(
                model_name="membrane-default",
                world_size=1,
                local_world_size=1,
                worker_id=0,
                local_worker_id=0,
                kv_dtype="fp16",
                kv_shape=(32, 1, 1, 128, 64),
            )
            self._backend = LocalCPUBackend(
                config=engine_config, metadata=metadata
            )
        except Exception as exc:
            raise RuntimeError(f"failed to initialise LMCache backend: {exc}") from exc
        self._keys: dict[int, str] = {}
        self._objects: dict[int, Any] = {}
        self._size = 0
        logger.info("LMCacheContentStore initialised")

    def _make_object(self, data: bytes) -> Any:
        """Wrap raw bytes in an LMCache ``MemoryObj``.

        Args:
            data: Canonical frame bytes.

        Returns:
            TensorMemoryObj: A ``(1, 1, 1, N)`` uint8 ``MemoryObj``
            ready for ``LocalCPUBackend.submit_put_task``.
        """
        import torch
        from lmcache.v1.memory_management import (
            MemoryFormat,
            MemoryObjMetadata,
            TensorMemoryObj,
        )

        if not data:
            tensor = torch.zeros((1, 1, 1, 1), dtype=torch.uint8)
        else:
            tensor = torch.frombuffer(
                bytearray(data), dtype=torch.uint8
            ).reshape(1, 1, 1, -1)
        meta = MemoryObjMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            address=0,
            phy_size=tensor.element_size() * tensor.numel(),
            ref_count=1,
            fmt=MemoryFormat.UNDEFINED,
        )
        # LMCache 0.5.4's TensorMemoryObj requires the
        # backend's memory allocator as its parent so the
        # ``__del__`` finalizer can free the slot. We pull it
        # through the backend's ``get_memory_allocator()`` (no
        # arguments; the shape/dtype are inferred from the
        # last ``allocate`` call).
        parent_allocator = self._backend.get_memory_allocator()
        return TensorMemoryObj(
            raw_data=tensor, metadata=meta, parent_allocator=parent_allocator
        )

    def _read_object(self, memory_obj: Any) -> bytes:
        """Read a ``MemoryObj`` back into raw bytes.

        Args:
            memory_obj: LMCache ``MemoryObj``.

        Returns:
            bytes: The original payload.
        """
        return bytes(memory_obj.tensor.reshape(-1).tolist())

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` via the LMCache backend.

        Args:
            key: Opaque key string.
            data: Canonical frame bytes.

        Raises:
            OSError: When LMCache's put raises.
        """
        lmcache_key = _hex_key_to_int(key)
        memory_obj = self._make_object(data)
        with self._lock:
            # ``LocalCPUBackend.submit_put_task`` returns a future;
            # we block on it so the ``ContentStore`` contract stays
            # synchronous.
            future = self._backend.submit_put_task(
                key=lmcache_key, memory_obj=memory_obj
            )
            if future is not None:
                future.result(timeout=30.0)
            self._objects[lmcache_key] = memory_obj
            self._keys[lmcache_key] = key
            self._size += len(data)

    def get(self, key: str) -> bytes | None:
        """Retrieve the bytes previously stored under ``key``.

        Args:
            key: Opaque key.

        Returns:
            bytes | None: Stored bytes, or ``None`` when absent.
        """
        lmcache_key = _hex_key_to_int(key)
        with self._lock:
            memory_obj = self._objects.get(lmcache_key)
        if memory_obj is None:
            return None
        return self._read_object(memory_obj)

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present in the LMCache backend.

        Args:
            key: Opaque key.

        Returns:
            bool: ``True`` when the backend has the bytes for ``key``.
        """
        with self._lock:
            return _hex_key_to_int(key) in self._objects

    def delete(self, key: str) -> bool:
        """Remove the entry at ``key``.

        Args:
            key: Opaque key.

        Returns:
            bool: ``True`` when a removal actually occurred.
        """
        lmcache_key = _hex_key_to_int(key)
        with self._lock:
            memory_obj = self._objects.pop(lmcache_key, None)
            self._backend.remove(lmcache_key)
            if memory_obj is not None:
                self._keys.pop(lmcache_key, None)
        return memory_obj is not None

    def size(self) -> int:
        """Return the total bytes currently held.

        Returns:
            int: Sum of ``len(data)`` across every entry.
        """
        return self._size


__all__ = ["LMCacheContentStore"]
