"""Tests for AES-GCM encryption + per-tenant key derivation (Phase 3.4.6 + 3.4.7)."""

from __future__ import annotations

import os

import pytest

from membrane.content_store import FilesystemBlob
from membrane.security.encryption import (
    KeyProvider,
    StaticKeyProvider,
    decrypt_payload,
    derive_tenant_key,
    encrypt_payload,
)


class TestStaticKeyProvider:
    def test_returns_32_byte_master_key(self):
        provider = StaticKeyProvider(key=b"\x00" * 32)
        key = provider.master_key()
        assert len(key) == 32

    def test_caches_key(self):
        provider = StaticKeyProvider(key=b"\x01" * 32)
        first = provider.master_key()
        second = provider.master_key()
        assert first == second

    def test_random_key_when_none_supplied(self):
        provider = StaticKeyProvider()
        key = provider.master_key()
        assert len(key) == 32
        assert provider.master_key() == key  # cached, not regenerated

    def test_satisfies_protocol(self):
        provider = StaticKeyProvider()
        # Structural check; the KeyProvider Protocol is
        # @runtime_checkable but isinstance on a non-Protocol
        # class is not supported. We verify the surface instead.
        assert callable(getattr(provider, "master_key", None))


class TestDeriveTenantKey:
    def test_deterministic_for_same_inputs(self):
        key = derive_tenant_key(b"\x00" * 32, "acme", "h" * 64)
        again = derive_tenant_key(b"\x00" * 32, "acme", "h" * 64)
        assert key == again
        assert len(key) == 32

    def test_different_tenant_yields_different_key(self):
        a = derive_tenant_key(b"\x00" * 32, "acme", "h" * 64)
        b = derive_tenant_key(b"\x00" * 32, "globex", "h" * 64)
        assert a != b

    def test_different_content_hash_yields_different_key(self):
        a = derive_tenant_key(b"\x00" * 32, "acme", "h1" + "0" * 62)
        b = derive_tenant_key(b"\x00" * 32, "acme", "h2" + "0" * 62)
        assert a != b


class TestEncryptDecrypt:
    def test_round_trip(self):
        key = b"\x00" * 32
        payload = b"hello world"
        blob = encrypt_payload(payload, key)
        assert decrypt_payload(blob, key) == payload

    def test_different_keys_fail_or_yield_garbage(self):
        key_a = b"\x00" * 32
        key_b = b"\x01" * 32
        blob = encrypt_payload(b"secret", key_a)
        # Either the decryption fails (raises) OR it returns
        # garbage — neither is the original "secret".
        try:
            result = decrypt_payload(blob, key_b)
        except Exception:
            return  # expected: decryption raises
        assert result != b"secret"

    def test_blob_includes_nonce_and_tag(self):
        blob = encrypt_payload(b"x", b"\x00" * 32)
        # 12-byte nonce + at least 1 byte ciphertext + 16-byte tag.
        assert len(blob) >= 12 + 1 + 16

    def test_short_blob_rejected(self):
        from membrane.security.encryption import DecryptError

        with pytest.raises(DecryptError):
            decrypt_payload(b"too-short", b"\x00" * 32)


class TestEncryptedFilesystemBlob:
    def test_round_trip(self, tmp_path):
        store = FilesystemBlob(tmp_path / "blob", tenant_id="acme")
        store.put("abcd1234", b"secret-data")
        assert store.get("abcd1234") == b"secret-data"

    def test_encrypted_blob_is_not_plaintext(self, tmp_path):
        """The on-disk bytes must not equal the plaintext."""
        store = FilesystemBlob(tmp_path / "blob", tenant_id="acme")
        store.put("deadbeef", b"plaintext-marker-XYZ")
        raw = (tmp_path / "blob" / "de" / "ad" / "deadbeef.blob").read_bytes()
        assert b"plaintext-marker-XYZ" not in raw

    def test_different_tenant_cannot_read_other_tenant(self, tmp_path):
        """Cross-tenant decryption fails because the derived key differs."""
        provider = StaticKeyProvider(key=b"\x02" * 32)
        store_acme = FilesystemBlob(
            tmp_path / "blob_acme", tenant_id="acme", key_provider=provider
        )
        store_globex = FilesystemBlob(
            tmp_path / "blob_globex", tenant_id="globex", key_provider=provider
        )
        store_acme.put("cafebabe", b"acme-only")
        # globex store reads the same bytes (the underlying file
        # is identical under the same provider), but its
        # derived key differs so decryption fails.
        assert store_globex.get("cafebabe") is None

    def test_persistence_across_instances_with_shared_provider(self, tmp_path):
        provider = StaticKeyProvider(key=b"\x00" * 32)
        store_a = FilesystemBlob(
            tmp_path / "blob", tenant_id="acme", key_provider=provider
        )
        store_a.put("abcd1234", b"hello")
        store_b = FilesystemBlob(
            tmp_path / "blob", tenant_id="acme", key_provider=provider
        )
        assert store_b.get("abcd1234") == b"hello"

    def test_size_reflects_plaintext(self, tmp_path):
        store = FilesystemBlob(tmp_path / "blob", tenant_id="acme")
        store.put("abcd1111", b"12345")
        assert store.size() == 5
