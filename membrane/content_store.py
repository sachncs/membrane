"""Canonical byte-store :class:`ContentStore` and two implementations.

The v1 storage layer round-trips fragments through a flat
metadata hash plus a separately keyed blob. Two physical
backends are provided:

* :class:`InProcessBytes` — ``dict[str, bytes]`` in memory, thread-safe,
  loses state on process exit. Default for tests and single-process
  deployments.

* :class:`FilesystemBlob` — atomic file writes under
  ``{root}/{xx}/{yy}/{key}.blob`` with fsync on commit. Survives
  process restart. Optional: requires ``os.replace`` + ``os.fsync``,
  both available on POSIX.

S3-backed object storage is intentionally not implemented here; the
deployment story is Redis (for metadata and directory) plus the local
filesystem (for the bytes themselves). Operators who need S3 can wrap
their own client as a third implementation behind the same
:class:`ContentStore` Protocol.

The Protocol:

* ``put(key, data)`` — overwrite atomically; idempotent.
* ``get(key)`` — ``bytes`` or ``None`` when missing.
* ``has(key)`` — membership check.
* ``delete(key)`` — best-effort; returns ``True`` when a removal
  actually happened.
* ``size()`` — total bytes currently held (cheap O(1) for both).

Key strings are opaque to the store. By convention
:meth:`membrane.canonical.canonicalize`'s
``identity.payload_hash`` is the key, but any string up to 256
characters is supported.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ContentStore(Protocol):
    """Storage for canonical fragment frames keyed by string.

    Implementations must be **idempotent on overwrite**: calling
    :meth:`put` with the same key twice must always return the
    latest bytes; concurrent puts with the same key from different
    threads must not interleave a partially written frame.
    """

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``.

        Args:
            key: Opaque key, conventionally the fragment's
                ``identity.payload_hash``.
            data: Canonical frame bytes.

        Raises:
            OSError: When the underlying backend cannot accept
                the write (e.g. out-of-disk space).
        """
        ...

    def get(self, key: str) -> bytes | None:
        """Retrieve the bytes previously stored under ``key``.

        Args:
            key: Key used in a prior :meth:`put`.

        Returns:
            bytes | None: The bytes, or ``None`` when absent.
        """
        ...

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present in the store.

        Args:
            key: Key to probe.

        Returns:
            bool: ``True`` when a value is stored under ``key``.
        """
        ...

    def delete(self, key: str) -> bool:
        """Remove the entry at ``key``.

        Args:
            key: Key to delete.

        Returns:
            bool: ``True`` when a removal actually occurred;
            ``False`` when the key was already absent.
        """
        ...

    def size(self) -> int:
        """Return the total number of bytes currently held.

        Returns:
            int: Sum of ``len(data)`` across every entry.
        """
        ...


class InProcessBytes:
    """Thread-safe in-memory :class:`ContentStore`.

    The default implementation used by tests and single-process
    deployments. State is lost on process exit; durability is
    the responsibility of a higher-tier store.

    Attributes:
        capacity_bytes: Optional ceiling. ``None`` for unbounded.
    """

    def __init__(self, capacity_bytes: int | None = None) -> None:
        """Initialize the in-process store.

        Args:
            capacity_bytes: Maximum bytes the store will retain
                before refusing new writes. ``None`` for unlimited.
        """
        self._store: dict[str, bytes] = {}
        self._lock = threading.RLock()
        self._used_bytes = 0
        self.capacity_bytes = capacity_bytes

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``.

        Args:
            key: Opaque key.
            data: Byte string.

        Raises:
            ValueError: When ``data`` exceeds ``capacity_bytes``
                (which records both cap and current usage).
        """
        with self._lock:
            if self.capacity_bytes is not None and len(data) > self.capacity_bytes:
                raise ValueError(
                    f"payload {len(data)} bytes exceeds capacity_bytes {self.capacity_bytes}"
                )
            self._store[key] = data
            self._used_bytes = sum(len(b) for b in self._store.values())

    def get(self, key: str) -> bytes | None:
        """Return the bytes stored under ``key`` or ``None``."""
        with self._lock:
            return self._store.get(key)

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present."""
        with self._lock:
            return key in self._store

    def delete(self, key: str) -> bool:
        """Remove and return whether the removal actually happened."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._used_bytes = sum(len(b) for b in self._store.values())
                return True
            return False

    def size(self) -> int:
        """Return total bytes currently held."""
        with self._lock:
            return self._used_bytes

    def __len__(self) -> int:
        """Return the number of distinct entries held."""
        with self._lock:
            return len(self._store)


class FilesystemBlob:
    """On-disk :class:`ContentStore` with two-level sharded layout.

    Layout::

        {root}/{key[:2]}/{key[2:4]}/{key}.blob

    The two-level prefix spreads the directory fanout across
    roughly 65k top-level directories. Writes go through a temp
    file in the same directory followed by ``os.replace`` which
    is atomic on POSIX. After the rename we ``os.fsync`` the
    parent directory so the rename is durable across a crash.

    Thread safety: the public methods are protected by a lock,
    but write atomicity also comes from the temp-file +
    ``os.replace`` flow, so a single-threaded writer is already
    crash-safe.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Initialize the on-disk store.

        Args:
            root: Directory under which blob files are written.
                Created when missing.

        Raises:
            OSError: When ``root`` cannot be created.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._used_bytes = 0

    def _path_for(self, key: str) -> Path:
        """Return the on-disk path for ``key``.

        Args:
            key: Opaque key (typically 64 hex chars).

        Returns:
            Path: ``{root}/{key[:2]}/{key[2:4]}/{key}.blob``.

        Raises:
            ValueError: When ``key`` is shorter than 4 characters,
                since the sharding scheme requires at least that
                much separation.
        """
        if len(key) < 4:
            raise ValueError(f"key must be at least 4 chars for sharded layout, got {key!r}")
        shard_a = key[:2]
        shard_b = key[2:4]
        directory = self.root / shard_a / shard_b
        return directory / f"{key}.blob"

    def put(self, key: str, data: bytes) -> None:
        """Atomically write ``data`` to ``key``.

        Args:
            key: Opaque key.
            data: Byte string.

        Raises:
            OSError: When the underlying filesystem rejects a
                write or rename.
        """
        target = self._path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=str(target.parent),
                prefix=f".{key}.",
                suffix=".tmp",
            ) as tmp:
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = tmp.name
            os.replace(tmp_path, target)
            # fsync the directory so the rename is durable across
            # process exit on POSIX.
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            self._used_bytes = self._walk_size()

    def get(self, key: str) -> bytes | None:
        """Read the bytes stored under ``key`` or ``None``."""
        path = self._path_for(key)
        if not path.exists():
            return None
        with self._lock:
            return path.read_bytes()

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present on disk."""
        return self._path_for(key).exists()

    def delete(self, key: str) -> bool:
        """Remove the blob at ``key``."""
        path = self._path_for(key)
        with self._lock:
            try:
                path.unlink()
                # Try to remove empty parent dirs to keep the
                # layout tidy; failures are non-fatal because
                # subsequent writes recreate them.
                with contextlib.suppress(OSError):
                    path.parent.rmdir()
                with contextlib.suppress(OSError):
                    path.parent.parent.rmdir()
                self._used_bytes = self._walk_size()
                return True
            except FileNotFoundError:
                return False

    def size(self) -> int:
        """Sum the byte size of every ``*.blob`` under :attr:`root`."""
        with self._lock:
            return self._used_bytes

    def _walk_size(self) -> int:
        """Walk the root and sum the size of every blob file."""
        total = 0
        for path in self.root.rglob("*.blob"):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def __len__(self) -> int:
        """Return the number of blob files under :attr:`root`."""
        return sum(1 for _ in self.root.rglob("*.blob"))


__all__ = [
    "ContentStore",
    "FilesystemBlob",
    "InProcessBytes",
]
