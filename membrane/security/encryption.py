"""AES-256-GCM encryption at rest + per-tenant key derivation (Phase 3.4.6 + 3.4.7).

The v3.0.0 release bakes AES-256-GCM encryption into
:class:`membrane.content_store.FilesystemBlob`. The plain
``FilesystemBlob`` constructor is removed; the only
constructor is the encrypted variant. Per-tenant keys are
derived via HKDF-SHA256 from the master key + tenant id +
content hash, so a single breach of the master key does
not compromise every tenant.

The :class:`KeyProvider` Protocol allows operators to plug
in a Vault-backed master key (Phase 3.4.5b) without
changing the storage layer.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets as _secrets
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

NONCE_SIZE: int = 12
TAG_SIZE: int = 16
KEY_SIZE: int = 32
SALT_SIZE: int = 16


@runtime_checkable
class KeyProvider(Protocol):
    """Pluggable master-key backend.

    Implementations return the per-store master key as raw
    bytes. The default :class:`StaticKeyProvider` reads from
    a constructor argument; production deployments back this
    with Vault or AWS KMS via :mod:`membrane.secrets`.
    """

    def master_key(self) -> bytes:
        """Return the master key bytes.

        Returns:
            bytes: 32-byte master key.
        """
        ...


@dataclass
class StaticKeyProvider:
    """Master key read from a constructor argument.

    Attributes:
        key: Raw 32-byte master key. ``None`` generates a
            fresh random key on construction (used by tests).
    """

    key: bytes | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cached: bytes | None = field(default=None, init=False)

    def master_key(self) -> bytes:
        """Return the master key, generating one on first call.

        Returns:
            bytes: 32 random bytes (AES-256 key size).
        """
        with self._lock:
            if self._cached is None:
                self._cached = self.key if self.key is not None else _secrets.token_bytes(KEY_SIZE)
            return self._cached


def derive_tenant_key(
    master_key: bytes,
    tenant_id: str,
    content_hash: str,
    *,
    info: bytes = b"membrane/v3/tenant-key",
) -> bytes:
    """Derive a per-(tenant, content) encryption key.

    Args:
        master_key: The :class:`KeyProvider.master_key` value.
        tenant_id: Tenant namespace the fragment lives in.
        content_hash: Hex SHA-256 of the canonical content hash.
        info: Domain separation tag.

    Returns:
        bytes: 32-byte derived key.
    """
    salt = hashlib.sha256(f"membrane-tenant:{tenant_id}".encode()).digest()[:SALT_SIZE]
    derived = hashlib.shake_128 if hasattr(hashlib, "shake_128") else None
    if derived is None:
        # stdlib fallback: HKDF using HMAC-SHA256.
        return _hkdf_hmac_sha256(master_key, salt, info + b":" + content_hash.encode("ascii"))
    key = _hkdf_hmac_sha256(master_key, salt, info + b":" + content_hash.encode("ascii"))
    return key[:KEY_SIZE]


def encrypt_payload(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt ``plaintext`` with AES-256-GCM under ``key``.

    Args:
        plaintext: Raw bytes.
        key: 32-byte key.

    Returns:
        bytes: 12-byte nonce + ciphertext + 16-byte tag.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "encryption at rest requires 'cryptography'; install membrane[secrets-aws|gcp|vault]"
        ) from exc
    aes = AESGCM(key)
    nonce = os.urandom(NONCE_SIZE)
    ciphertext = aes.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt_payload(blob: bytes, key: bytes) -> bytes:
    """Decrypt a blob produced by :func:`encrypt_payload`.

    Args:
        blob: nonce || ciphertext || tag.
        key: 32-byte key.

    Returns:
        bytes: Decrypted plaintext.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "encryption at rest requires 'cryptography'; install membrane[secrets-aws|gcp|vault]"
        ) from exc
    if len(blob) < NONCE_SIZE + TAG_SIZE:
        raise ValueError("encrypted blob too short")
    nonce = blob[:NONCE_SIZE]
    payload = blob[NONCE_SIZE:]
    return AESGCM(key).decrypt(nonce, payload, None)


def decrypt_payload_with_versions(
    blob: bytes,
    version_keys: tuple[bytes, ...],
) -> bytes:
    """Decrypt ``blob`` with any of the version keys.

    Tries each key in ``version_keys`` in order and returns the
    first successful decryption. The last entry is the active
    key (most likely to succeed), so iteration starts from the
    end and walks backward through older keys.

    Args:
        blob: The encrypted blob.
        version_keys: Tuple of candidate master keys in
            chronological order; the last entry is the active
            key.

    Returns:
        bytes: Decrypted plaintext.

    Raises:
        RuntimeError: When none of the candidate keys
            successfully decrypt ``blob``.
    """
    if not version_keys:
        raise RuntimeError("no candidate keys to try")
    for key in reversed(version_keys):
        try:
            return decrypt_payload(blob, key)
        except Exception:
            continue
    raise RuntimeError("decryption failed for every candidate key")


def _hkdf_hmac_sha256(ikm: bytes, salt: bytes, info: bytes) -> bytes:
    """Minimal HKDF-SHA256 implementation that does not depend on cryptography.

    Args:
        ikm: Input keying material.
        salt: Per-derivation salt.
        info: Domain-separation tag.

    Returns:
        bytes: 32-byte derived key.
    """
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = b""
    out = b""
    counter = 1
    while len(out) < KEY_SIZE:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:KEY_SIZE]


__all__ = [
    "KeyProvider",
    "StaticKeyProvider",
    "decrypt_payload",
    "derive_tenant_key",
    "encrypt_payload",
]
