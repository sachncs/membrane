"""Cross-tenant encryption integration test (Phase 3.4.6 + 3.4.7 follow-up).

The v3.0.0 release encrypts every per-(tenant, content_hash)
entry with an AES-256-GCM key derived from the master key +
the tenant id. This test exercises the full cross-tenant
isolation flow:

* Tenant A writes a fragment.
* Tenant B (different derived key) cannot decrypt it.
* The same tenant + same master key can.
* Rotating the master key to v2 (via RotatingKeyProvider)
  preserves the ability to decrypt legacy v1 ciphertexts.
"""

from __future__ import annotations

import pytest

from membrane.content_store_encrypted import EncryptedInProcessBytes
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.security.encryption import StaticKeyProvider
from membrane.security.key_rotation import RotatingKeyProvider


def _fragment(tenant: str, payload_size: int = 10) -> Fragment:
    """Build a fragment for a given tenant."""
    ident = PayloadIdentity(
        payload_hash=f"hash-{tenant}".ljust(64, "0")[:64],
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 1),
        dtype="float16",
        shape=(1, 1, 1, 1, 64),
    )
    return Fragment(
        identity=ident,
        payload_ref=ident.payload_hash,
        payload_size=payload_size,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
        tenant_id=tenant,
    )


class TestCrossTenantEncryption:
    def test_tenant_b_cannot_decrypt_tenant_a(self):
        provider = StaticKeyProvider(key=b"\x00" * 32)
        store_a = EncryptedInProcessBytes(tenant_id="acme", key_provider=provider)

        # Tenant A stores a payload.
        store_a.put("k", b"acme-secret")

        # Inspect the on-disk ciphertext from A.
        ciphertext = store_a._store["k"]  # type: ignore[attr-defined]
        # Tenant B can read the bytes from its dict (since this
        # is a single-process test the dicts are separate) but
        # cannot decrypt them: the derived key differs.
        # We simulate the cross-tenant read by giving B the
        # ciphertext and trying to decrypt.
        from membrane.security.encryption import (
            DecryptError,
            decrypt_payload,
            derive_tenant_key,
        )

        a_key = derive_tenant_key(b"\x00" * 32, "acme", "k")
        b_key = derive_tenant_key(b"\x00" * 32, "globex", "k")
        # A's key decrypts.
        assert decrypt_payload(ciphertext, a_key) == b"acme-secret"
        # B's key fails.
        with pytest.raises(DecryptError):
            decrypt_payload(ciphertext, b_key)

    def test_tenant_a_can_decrypt_with_its_own_key(self):
        provider = StaticKeyProvider(key=b"\x00" * 32)
        store_a = EncryptedInProcessBytes(tenant_id="acme", key_provider=provider)
        store_a.put("k", b"acme-data")
        # Use the same provider to derive the same key; the
        # ciphertext is decryptable.
        from membrane.security.encryption import (
            decrypt_payload,
            derive_tenant_key,
        )

        k = derive_tenant_key(b"\x00" * 32, "acme", "k")
        ciphertext = store_a._store["k"]  # type: ignore[attr-defined]
        assert decrypt_payload(ciphertext, k) == b"acme-data"

    def test_master_key_rotation_preserves_tenant_isolation(self):
        """v1 -> v2 rotation does not leak across tenants."""
        # Phase 1: write with v1.
        old_provider = RotatingKeyProvider(initial_key=b"\x00" * 32)
        store = EncryptedInProcessBytes(tenant_id="acme", key_provider=old_provider)
        store.put("acme-k", b"acme-v1")

        # Phase 2: rotate to v2.
        old_provider.rotate(b"\x01" * 32)

        # Same store decrypts acme-v1 (tenant a, v1 key derived).
        assert store.get("acme-k") == b"acme-v1"
        # The derived keys for tenant b differ even after rotation.
        from membrane.security.encryption import derive_tenant_key

        a_v1 = derive_tenant_key(b"\x00" * 32, "acme", "acme-k")
        a_v2 = derive_tenant_key(b"\x01" * 32, "acme", "acme-k")
        b_v1 = derive_tenant_key(b"\x00" * 32, "globex", "acme-k")
        b_v2 = derive_tenant_key(b"\x01" * 32, "globex", "acme-k")
        # A v1 != A v2 (rotation changes the master key).
        assert a_v1 != a_v2
        # A != B (rotation does not break tenant isolation).
        assert a_v1 != b_v1
        assert a_v2 != b_v2

    def test_audit_log_integration_with_tenant_filter(self):
        """The audit log records actor = tenant; queries filter by it."""
        from membrane.audit import AuditLog

        log = AuditLog()
        log.record(actor="acme", action="fragment.store", payload={"hash": "h1"})
        log.record(actor="globex", action="fragment.store", payload={"hash": "h2"})
        log.record(actor="acme", action="fragment.retrieve", payload={"hash": "h1"})
        # Filter by tenant (actor = tenant) and verify.
        acme_entries = [e for e in log.all() if e.actor == "acme"]
        assert len(acme_entries) == 2
        # The chain verifies.
        from membrane.audit import verify_chain

        assert verify_chain(log.all()) is None
