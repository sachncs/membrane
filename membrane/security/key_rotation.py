"""Master-key rotation for :class:`KeyProvider` (Phase 3.4.6 follow-up).

The v3.0.0 release adds :class:`RotatingKeyProvider`, a
:class:`KeyProvider` that keeps a versioned list of master
keys. New encryptions use the active key; decryptions
transparently fall back to older versions when the active
key fails, which lets operators roll the master key
without re-encrypting the entire on-disk store.

The helper is opt-in: production deployments continue to
use :class:`StaticKeyProvider` for a single-key setup, and
adopt :class:`RotatingKeyProvider` when they need a rotation
cycle.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from membrane.security.encryption import KeyProvider

logger = logging.getLogger(__name__)


@dataclass
class KeyVersion:
    """One master-key version.

    Attributes:
        version: Monotonically increasing version id.
        key: Raw 32-byte master key.
    """

    version: int
    key: bytes


class RotatingKeyProvider(KeyProvider):
    """A versioned master-key store.

    The v3.0.0 release ships the rotation primitive so a
    cluster can roll a new master key without re-encrypting
    every fragment: the new key becomes active for new
    encryptions while old ciphertexts stay decryptable with
    their original key.

    Attributes:
        key_versions: Ordered list of master keys; the last
            entry is the active key.
    """

    def __init__(self, initial_key: bytes) -> None:
        """Initialize the provider with a single version.

        Args:
            initial_key: 32-byte master key for v1.

        Raises:
            ValueError: When ``initial_key`` is not 32 bytes.
        """
        if len(initial_key) != 32:
            raise ValueError(
                f"initial_key must be 32 bytes, got {len(initial_key)}"
            )
        self._lock = threading.RLock()
        self._versions: list[KeyVersion] = [KeyVersion(version=1, key=initial_key)]
        self._active: bytes = initial_key

    @property
    def active_version(self) -> int:
        """Return the active version id.

        Returns:
            int: The currently active key version.
        """
        with self._lock:
            return self._versions[-1].version

    def rotate(self, new_key: bytes) -> int:
        """Add ``new_key`` as the active master key.

        Args:
            new_key: 32-byte replacement master key.

        Returns:
            int: The new active version id.

        Raises:
            ValueError: When ``new_key`` is not 32 bytes.
        """
        if len(new_key) != 32:
            raise ValueError(
                f"new_key must be 32 bytes, got {len(new_key)}"
            )
        with self._lock:
            new_version = self._versions[-1].version + 1
            self._versions.append(KeyVersion(version=new_version, key=new_key))
            self._active = new_key
            logger.info(
                "RotatingKeyProvider rotated to version %d (kept %d older versions)",
                new_version,
                len(self._versions) - 1,
            )
            return new_version

    def version_keys(self) -> tuple[bytes, ...]:
        """Return the current version keys in version order.

        Returns:
            tuple[bytes, ...]: The master keys in chronological
            order; the last entry is the active key.
        """
        with self._lock:
            return tuple(v.key for v in self._versions)

    def master_key(self) -> bytes:
        """Return the active master key.

        Returns:
            bytes: The currently active 32-byte master key.
        """
        with self._lock:
            return self._active


__all__ = ["KeyVersion", "RotatingKeyProvider"]
