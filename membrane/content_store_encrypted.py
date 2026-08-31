"""Encrypted in-memory content store (Phase 3.4.6 follow-up).

The v3.0.0 release ships :class:`EncryptedInProcessBytes`, an
in-memory variant of :class:`membrane.content_store.FilesystemBlob`
that encrypts every payload with the per-(tenant, content_hash)
AES-256-GCM key derived from the same :class:`KeyProvider`
the on-disk variant uses. The class is intended for
single-process deployments that need at-rest encryption
without the file-system layout (CI, ephemeral workloads,
sidecar containers).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

from membrane.security.encryption import (
    KeyProvider,
    StaticKeyProvider,
    decrypt_payload,
    derive_tenant_key,
    encrypt_payload,
)


class EncryptedInProcessBytes:
    """Thread-safe encrypted in-memory content store.

    Implements the same contract as
    :class:`membrane.content_store.InProcessBytes` /
    :class:`membrane.content_store.FilesystemBlob` (the parts
    used by ``Node``) but encrypts every put / get with
    AES-256-GCM. The class is not a :class:`ContentStore`
    subclass (the v3.0.0 release removes the base class
    inheritance) -- it mirrors the same methods.
    """

    def __init__(
        self,
        tenant_id: str,
        key_provider: KeyProvider | None = None,
        *,
        capacity_bytes: int | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            tenant_id: Tenant namespace the store keeps data
                on behalf of. Different tenants get different
                derived keys (Phase 3.4.7).
            key_provider: Optional :class:`KeyProvider`. When
                ``None``, a :class:`StaticKeyProvider` is
                constructed and a fresh random master key is
                generated; production deployments back this
                with a Vault or AWS KMS secret backend via
                :mod:`membrane.secrets`.
            capacity_bytes: Optional byte cap. ``None`` is
                unlimited.

        Raises:
            ValueError: When ``capacity_bytes`` is negative.
        """
        if capacity_bytes is not None and capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        self.tenant_id = tenant_id
        self.capacity_bytes = capacity_bytes
        self._provider = key_provider or StaticKeyProvider()
        self._master_key = self._provider.master_key()
        self._store: dict[str, bytes] = {}
        self._used_bytes = 0
        self._lock = threading.RLock()

    def put(self, key: str, data: bytes) -> None:
        """Encrypt and store ``data`` under ``key``.

        Args:
            key: Opaque key.
            data: Plaintext bytes.

        Raises:
            ValueError: When the store would exceed its
                capacity cap.
        """
        per_key = derive_tenant_key(self._master_key, self.tenant_id, key)
        blob = encrypt_payload(data, per_key)
        with self._lock:
            if self.capacity_bytes is not None and self._used_bytes + len(blob) > self.capacity_bytes:
                raise ValueError("capacity exceeded")
            self._store[key] = blob
            # ``_used_bytes`` tracks the plaintext size so the
            # operator-visible accounting matches
            # FilesystemBlob's surface (Phase 3.4.6 + 3.4.7).
            self._used_bytes += len(data)

    def delete(self, key: str) -> bool:
        """Remove the entry at ``key``.

        Args:
            key: Opaque key.

        Returns:
            bool: True when the key was present.
        """
        with self._lock:
            existing = self._store.pop(key, None)
            if existing is None:
                return False
            # We did not store the plaintext length separately;
            # remove the same blob size from the running total.
            self._used_bytes = max(0, self._used_bytes - (len(existing) - 28))
            return True

    def get(self, key: str) -> bytes | None:
        """Decrypt and return the bytes stored under ``key``.

        Args:
            key: Opaque key.

        Returns:
            bytes | None: Decrypted bytes or ``None`` when the
            key is absent or the decryption fails (the latter
            looks identical to an absent key for the caller).
        """
        with self._lock:
            blob = self._store.get(key)
        if blob is None:
            return None
        per_key = derive_tenant_key(self._master_key, self.tenant_id, key)
        try:
            return decrypt_payload(blob, per_key)
        except Exception:
            return None

    def has(self, key: str) -> bool:
        """Return True when ``key`` is present on disk.

        Args:
            key: Opaque key.

        Returns:
            bool: Presence flag.
        """
        with self._lock:
            return key in self._store

    def size(self) -> int:
        """Return the plaintext byte total.

        Returns:
            int: Sum of decrypted plaintext sizes.
        """
        with self._lock:
            return self._used_bytes

    def __len__(self) -> int:
        """Return the entry count.

        Returns:
            int: ``len(self._store)``.
        """
        with self._lock:
            return len(self._store)

    def __iter__(self) -> Iterator[str]:
        """Iterate over stored keys.

        Returns:
            Iterator[str]: Keys in insertion order.
        """
        with self._lock:
            return iter(list(self._store.keys()))


__all__ = ["EncryptedInProcessBytes"]
