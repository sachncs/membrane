"""Encrypted in-memory content store.

The v3.0.0 release ships :class:`EncryptedInProcessBytes`, an
in-memory variant of :class:`membrane.content_store.FilesystemBlob`
that encrypts every payload with the per-(tenant, content_hash)
AES-256-GCM key derived from the same :class:`KeyProvider`
the on-disk variant uses. The class is intended for
single-process deployments that need at-rest encryption
without the file-system layout (CI, ephemeral workloads,
sidecar containers).
"""

import threading
from collections.abc import Iterator
from typing import override

from membrane.content_store import ContentStore
from membrane.security.encryption import (
    KeyProvider,
    StaticKeyProvider,
    decrypt_payload,
    derive_tenant_key,
    encrypt_payload,
)


class EncryptedInProcessBytes(ContentStore):
    """Thread-safe encrypted in-memory :class:`ContentStore`.

    Mirrors the on-disk :class:`membrane.content_store.FilesystemBlob`
    interface but stores ciphertexts in a Python dict. Useful
    for single-process deployments that need at-rest encryption
    without the file-system layout (CI, sidecar containers,
    ephemeral workloads).
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
                derived keys.
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
        self.__provider = key_provider or StaticKeyProvider()
        self.__master_key = self.__provider.master_key()
        self.store: dict[str, bytes] = {}
        self.__used_bytes = 0
        self.lock = threading.RLock()

    @override
    def put(self, key: str, data: bytes) -> None:
        """Encrypt and store ``data`` under ``key``.

        Args:
            key: Opaque key.
            data: Plaintext bytes.

        Raises:
            ValueError: When the store would exceed its
                capacity cap.
        """
        per_key = derive_tenant_key(self.__master_key, self.tenant_id, key)
        blob = encrypt_payload(data, per_key)
        with self.lock:
            if self.capacity_bytes is not None and self.__used_bytes + len(blob) > self.capacity_bytes:
                raise ValueError("capacity exceeded")
            self.store[key] = blob
            # ``__used_bytes`` tracks the plaintext size so the
            # operator-visible accounting matches
            # FilesystemBlob's surface.
            self.__used_bytes += len(data)

    @override
    def delete(self, key: str) -> bool:
        """Remove the entry at ``key``.

        Args:
            key: Opaque key.

        Returns:
            bool: True when the key was present.
        """
        with self.lock:
            existing = self.store.pop(key, None)
            if existing is None:
                return False
            # We did not store the plaintext length separately;
            # remove the same blob size from the running total.
            self.__used_bytes = max(0, self.__used_bytes - (len(existing) - 28))
            return True

    @override
    def get(self, key: str) -> bytes | None:
        """Decrypt and return the bytes stored under ``key``.

        Args:
            key: Opaque key.

        Returns:
            bytes | None: Decrypted bytes or ``None`` when the
            key is absent or the decryption fails (the latter
            looks identical to an absent key for the caller).
        """
        with self.lock:
            blob = self.store.get(key)
        if blob is None:
            return None
        # Try the active key first, then walk older version keys
        # so a RotatingKeyProvider can roll without re-encrypting.
        from membrane.security.encryption import (
            decrypt_payload_with_versions,
        )

        version_keys = getattr(self.__provider, "version_keys", None)
        if version_keys is not None:
            tenant_keys = tuple(derive_tenant_key(k, self.tenant_id, key) for k in version_keys())
            try:
                return decrypt_payload_with_versions(blob, tenant_keys)
            except RuntimeError:
                return None
        per_key = derive_tenant_key(self.__master_key, self.tenant_id, key)
        try:
            return decrypt_payload(blob, per_key)
        except Exception:
            return None

    @override
    def has(self, key: str) -> bool:
        """Return True when ``key`` is present on disk.

        Args:
            key: Opaque key.

        Returns:
            bool: Presence flag.
        """
        with self.lock:
            return key in self.store

    @override
    def size(self) -> int:
        """Return the plaintext byte total.

        Returns:
            int: Sum of decrypted plaintext sizes.
        """
        with self.lock:
            return self.__used_bytes

    def __len__(self) -> int:
        """Return the entry count.

        Returns:
            int: ``len(self.__store)``.
        """
        with self.lock:
            return len(self.store)

    def __iter__(self) -> Iterator[str]:
        """Iterate over stored keys.

        Returns:
            Iterator[str]: Keys in insertion order.
        """
        with self.lock:
            return iter(list(self.store.keys()))


__all__ = ["EncryptedInProcessBytes"]
