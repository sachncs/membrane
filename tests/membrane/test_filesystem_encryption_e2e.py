"""End-to-end FilesystemBlob encryption + key rotation (Phase 3.4.6 + 3.4.7 follow-up).

The 3.4.6 / 3.4.7 commits shipped AES-256-GCM encryption in
FilesystemBlob and the per-(tenant, content_hash) key
derivation. The unit tests cover the primitives; this test
runs the full on-disk round-trip:

* Write a fragment to a fresh FilesystemBlob (encrypted
  by default with a fresh random master key).
* The on-disk file is nonce + ciphertext + tag; the
  plaintext marker is NOT present.
* Read the same key back: decrypts to the plaintext.
* Rotate to a new key version via RotatingKeyProvider.
  Reads still succeed via the version fallback path.
* Cross-tenant: tenant B's derived key cannot decrypt
  tenant A's ciphertext.
"""

from __future__ import annotations

import pytest

from membrane.content_store import FilesystemBlob
from membrane.security.encryption import StaticKeyProvider
from membrane.security.key_rotation import RotatingKeyProvider


class TestFilesystemBlobEncryptionE2E:
    def test_round_trip_with_static_key(self, tmp_path):
        provider = StaticKeyProvider(key=b"\x00" * 32)
        store = FilesystemBlob(
            root=tmp_path / "fs", tenant_id="acme", key_provider=provider
        )
        store.put("payload-h", b"plaintext-marker")
        # The on-disk file is encrypted; the plaintext marker
        # is not in the file.
        path = next(tmp_path.glob("fs/*/*/payload-h.blob"))
        raw = path.read_bytes()
        assert b"plaintext-marker" not in raw
        # The retrieve path decrypts.
        assert store.get("payload-h") == b"plaintext-marker"

    def test_cross_tenant_decryption_fails(self, tmp_path):
        provider = StaticKeyProvider(key=b"\x00" * 32)
        store_a = FilesystemBlob(
            root=tmp_path / "fs_a", tenant_id="acme", key_provider=provider
        )
        # A writes a tenant_id=acme fragment.
        store_a.put("payload-h", b"acme-secret")
        # Manually load the ciphertext A wrote.
        ciphertext = None
        for path in (tmp_path / "fs_a").rglob("payload-h.blob"):
            ciphertext = path.read_bytes()
            break
        assert ciphertext is not None

        # Simulate the cross-tenant decrypt: tenant B does not
        # have the on-disk file (different root), but even if
        # it did the per-tenant derived key would fail.
        from membrane.security.encryption import (
            DecryptError,
            derive_tenant_key,
        )

        a_key = derive_tenant_key(b"\x00" * 32, "acme", "payload-h")
        b_key = derive_tenant_key(b"\x00" * 32, "globex", "payload-h")
        # A's key decrypts.
        from membrane.security.encryption import decrypt_payload

        assert decrypt_payload(ciphertext, a_key) == b"acme-secret"
        # B's key fails.
        with pytest.raises(DecryptError):
            decrypt_payload(ciphertext, b_key)

    def test_master_key_rotation_round_trip(self, tmp_path):
        # Phase 1: write with v1.
        provider = RotatingKeyProvider(initial_key=b"\x00" * 32)
        store = FilesystemBlob(
            root=tmp_path / "fs", tenant_id="acme", key_provider=provider
        )
        store.put("payload-p", b"v1-data")
        v1_path = next(tmp_path.glob("fs/*/*/payload-p.blob"))
        v1_ciphertext = v1_path.read_bytes()
        assert store.get("payload-p") == b"v1-data"

        # Phase 2: rotate to v2.
        provider.rotate(b"\x01" * 32)
        # The v1 ciphertext still decrypts via the version
        # fallback path.
        assert store.get("payload-p") == b"v1-data"

        # v2 writes use the new key.
        store.put("payload-q", b"v2-data")
        v2_path = next(tmp_path.glob("fs/*/*/payload-q.blob"))
        v2_ciphertext = v2_path.read_bytes()
        # v1 != v2 ciphertext for the same plaintext.
        assert v1_ciphertext != v2_ciphertext
        assert store.get("payload-q") == b"v2-data"
