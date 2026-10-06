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

import contextlib
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from membrane.security.encryption import KeyProvider, StaticKeyProvider

logger = logging.getLogger(__name__)


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
        self.store: dict[str, bytes] = {}
        self.lock = threading.RLock()
        self.__used_bytes = 0
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
        with self.lock:
            if self.capacity_bytes is not None and len(data) > self.capacity_bytes:
                raise ValueError(f"payload {len(data)} bytes exceeds capacity_bytes {self.capacity_bytes}")
            previous = self.store.get(key)
            self.store[key] = data
            self.__used_bytes += len(data) - (len(previous) if previous is not None else 0)

    def get(self, key: str) -> bytes | None:
        """Return the bytes stored under ``key`` or ``None``.

        Args:
            key: Opaque key.

        Returns:
            bytes | None: The bytes stored under ``key`` or ``None``.
        """
        return self.store.get(key)  # a single dict read: safe without the lock

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present.

        Args:
            key: Opaque key.

        Returns:
            bool: Whether ``key`` is present.
        """
        return key in self.store

    def delete(self, key: str) -> bool:
        """Remove and return whether the removal actually happened.

        Args:
            key: Opaque key.

        Returns:
            bool: True when an entry was removed.
        """
        with self.lock:
            removed = self.store.pop(key, None)
            if removed is None:
                return False
            self.__used_bytes -= len(removed)
            return True

    def size(self) -> int:
        """Return total bytes currently held.

        Returns:
            int: Total bytes currently held.
        """
        with self.lock:
            return self.__used_bytes

    def __len__(self) -> int:
        """Return the number of distinct entries held.

        Returns:
            int: The number of distinct entries held.
        """
        with self.lock:
            return len(self.store)


class FilesystemBlob:
    """On-disk :class:`ContentStore` with AES-256-GCM encryption.

    Layout::

        {root}/{key[:2]}/{key[2:4]}/{key}.blob

    The two-level prefix spreads the directory fanout across
    roughly 65k top-level directories. Writes go through a
    temp file in the same directory followed by ``os.replace``
    which is atomic on POSIX; the parent directory is
    ``fsync``-ed so the rename is durable across a crash.

    Every put is encrypted with AES-256-GCM under a per-
    (tenant, content_hash) derived key. Plain
    writes are not exposed; the v3.0.0 release drops the
    plaintext ``FilesystemBlob`` constructor in favor of
    ``FilesystemBlob(root, tenant_id, key_provider)``.

    Thread safety: the public methods are protected by a lock,
    but write atomicity also comes from the temp-file +
    ``os.replace`` flow, so a single-threaded writer is
    already crash-safe.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        tenant_id: str,
        key_provider: KeyProvider | None = None,
    ) -> None:
        """Initialize the encrypted on-disk store.

        Args:
            root: Directory under which blob files are
                written. Created when missing.
            tenant_id: Tenant namespace the store keeps files
                on behalf of. Different tenants get different
                derived keys.
            key_provider: Optional :class:`KeyProvider`. When
                ``None``, a :class:`StaticKeyProvider` is
                constructed and a fresh random master key is
                generated; production deployments back this
                with a Vault or AWS KMS secret backend via
                :mod:`membrane.secrets`.

        Raises:
            OSError: When ``root`` cannot be created.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tenant_id = tenant_id
        self.lock = threading.RLock()
        self.__used_bytes = 0
        self.__plaintext_bytes: dict[str, int] = {}
        self.__key_provider: KeyProvider = key_provider or StaticKeyProvider()

    def __path_for(self, key: str) -> Path:
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
        """Atomically write ``data`` (encrypted) to ``key``.

        Args:
            key: Opaque key.
            data: Byte string (encrypted before write).

        Raises:
            OSError: When the underlying filesystem rejects a
                write or rename.
        """
        from membrane.security.encryption import (
            derive_tenant_key,
            encrypt_payload,
        )

        target = self.__path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Ask the provider each time: a rotated key takes effect at once.
        per_key = derive_tenant_key(self.__key_provider.master_key(), self.tenant_id, key)
        blob = encrypt_payload(data, per_key)
        with self.lock:
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=str(target.parent),
                prefix=f".{key}.",
                suffix=".tmp",
            ) as tmp:
                tmp.write(blob)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = tmp.name
            os.replace(tmp_path, target)
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            self.__plaintext_bytes[key] = len(data)
            self.__used_bytes = sum(self.__plaintext_bytes.values())

    @property
    def key_provider(self) -> KeyProvider:
        """The master-key provider (a keyring when keys are versioned)."""
        return self.__key_provider

    def put_from_file(self, key: str, source_path: str) -> None:
        """Store the contents of ``source_path`` under ``key`` (encrypted).

        Args:
            key: Opaque key.
            source_path: Filesystem path of the source file.

        Raises:
            OSError: When the source cannot be read or the write fails.
        """
        self.put(key, Path(source_path).read_bytes())

    def keys(self) -> list[str]:
        """Return every key stored on disk.

        Returns:
            list[str]: Keys, in no particular order.
        """
        return [path.name.removesuffix(".blob") for path in self.root.rglob("*.blob")]

    def reencrypt_all(self) -> int:
        """Rewrite every blob not encrypted under the active master key.

        After a key rotation, old blobs stay readable through the older
        key versions; this pass moves them to the active key so the old
        versions can be retired.

        Returns:
            int: Blobs rewritten.
        """
        from membrane.security.encryption import decrypt_payload, derive_tenant_key

        rewritten = 0
        for key in self.keys():
            path = self.__path_for(key)
            try:
                with self.lock:
                    blob = path.read_bytes()
            except FileNotFoundError:
                continue
            active = derive_tenant_key(self.__key_provider.master_key(), self.tenant_id, key)
            try:
                decrypt_payload(blob, active)
                continue  # already under the active key
            except Exception:
                pass
            data = self.get(key)
            if data is None:
                logger.warning("blob %s cannot be decrypted with any key version; left as is", key)
                continue
            self.put(key, data)
            rewritten += 1
        return rewritten

    def get(self, key: str) -> bytes | None:
        """Read and decrypt the bytes stored under ``key``.

        Args:
            key: Opaque key.

        Returns:
            bytes | None: Decrypted bytes or ``None`` when the
            file is absent or the key derivation / decryption
            fails.

        Raises:
            OSError: When the underlying filesystem rejects the
                read.
        """
        from membrane.security.encryption import (
            decrypt_payload,
            derive_tenant_key,
        )

        path = self.__path_for(key)
        if not path.exists():
            return None
        with self.lock:
            blob = path.read_bytes()
        # Walk the version keys in reverse order so the active
        # key is tried first; older keys decrypt legacy blobs.
        from membrane.security.encryption import (
            decrypt_payload_with_versions,
        )

        version_keys = getattr(self.__key_provider, "version_keys", None)
        if version_keys is not None:
            tenant_keys = tuple(derive_tenant_key(k, self.tenant_id, key) for k in version_keys())
            try:
                return decrypt_payload_with_versions(blob, tenant_keys)
            except RuntimeError:
                return None
        per_key = derive_tenant_key(self.__key_provider.master_key(), self.tenant_id, key)
        try:
            return decrypt_payload(blob, per_key)
        except Exception:
            return None

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present on disk.

        Args:
            key: Opaque key.

        Returns:
            bool: Whether ``key`` is present on disk.
        """
        return self.__path_for(key).exists()

    def delete(self, key: str) -> bool:
        """Remove the blob at ``key``.

        Args:
            key: Opaque key.

        Returns:
            bool: True when a blob file was removed.
        """
        path = self.__path_for(key)
        with self.lock:
            try:
                path.unlink()
                # Try to remove empty parent dirs to keep the
                # layout tidy; failures are non-fatal because
                # subsequent writes recreate them.
                with contextlib.suppress(OSError):
                    path.parent.rmdir()
                with contextlib.suppress(OSError):
                    path.parent.parent.rmdir()
                self.__plaintext_bytes.pop(key, None)
                self.__used_bytes = sum(self.__plaintext_bytes.values())
                return True
            except FileNotFoundError:
                return False

    def size(self) -> int:
        """Sum the byte size of every ``*.blob`` under :attr:`root`.

        Returns:
            int: Plaintext bytes written through this store.
        """
        with self.lock:
            return self.__used_bytes

    def __walk_size(self) -> int:
        """Walk the root and sum the size of every blob file.

        Returns:
            int: Total size in bytes of the blob files on disk.
        """
        total = 0
        for path in self.root.rglob("*.blob"):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def __len__(self) -> int:
        """Return the number of blob files under :attr:`root`.

        Returns:
            int: The number of blob files under :attr:`root`.
        """
        return sum(1 for _ in self.root.rglob("*.blob"))


__all__ = [
    "ContentStore",
    "FilesystemBlob",
    "InProcessBytes",
]
