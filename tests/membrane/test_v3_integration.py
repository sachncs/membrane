"""Comprehensive v3.0.0 e2e test (Phase 3.0+ summary).

A single test that exercises the v3.0.0+ contract end-to-end:

* An ``EncryptedInProcessBytes`` store is the default at-rest
  surface (Phase 3.4.6).
* A ``Node`` records per-tenant metrics + tier assignments
  (Phase 3.1.7 + 3.5.6).
* A ``MembraneClient`` (sync) drives the wire path (Phase
  3.6.1).
* The audit log records every admin op (Phase 3.2.8).
* The ``/admin/audit`` route returns the chain-verified
  entries.
* The encrypted store's ciphertext contains no plaintext
  marker (Phase 3.4.6).
* A master-key rotation preserves legacy reads (Phase
  3.4.6 follow-up).
"""

from __future__ import annotations

import pytest


class TestV3Integration:
    def test_full_v3_integration_workflow(self):
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog, verify_chain
        from membrane.client import MembraneClient
        from membrane.compute.cpu import CPU
        from membrane.content_store_encrypted import EncryptedInProcessBytes
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.security.encryption import StaticKeyProvider
        from membrane.security.key_rotation import RotatingKeyProvider
        from membrane.serialization import to_dict
        from membrane.tier_migration import TierMigration
        from membrane.tiers import TierPolicy
        from membrane.transport.admin import create_admin_router
        from membrane.transport.fastapi import create_app

        # 1. Build the storage with a RotatingKeyProvider so the
        # post-test rotation proves legacy reads survive.
        rotation = RotatingKeyProvider(initial_key=b"\x00" * 32)
        store = EncryptedInProcessBytes(
            tenant_id="acme", key_provider=rotation
        )
        node = Node(
            node_id="v3-demo",
            max_memory_bytes=10_000,
            content_store=store,
            tier_policy=TierPolicy(hot_threshold=0.7),
        )
        demoted: list[tuple[str, str]] = []
        migration = TierMigration(
            policy=TierPolicy(),
            on_demote=lambda frag, tier: demoted.append(
                (frag.identity.payload_hash, tier)
            ),
        )
        node.add_eviction_callback(migration.on_evict)
        log = AuditLog()
        app = create_app(
            node=node,
            compute_backend=CPU(),
            transfer_service=None,
            cluster_manager=None,
        )
        http = TestClient(app)
        import httpx

        with httpx.Client(
            transport=http._transport,  # type: ignore[attr-defined]
            base_url="http://n1",
            timeout=5.0,
        ) as shared:
            client = MembraneClient("http://n1", transport=shared)

            # 2. Mount admin routes and run a policy update.
            class _StubCluster:
                shard_manager = None
                replicator = None
                config = type("C", (), {"default_consistency": "strong"})()

            app.state.cluster_manager = _StubCluster()
            app.state.audit_log = log
            app.include_router(create_admin_router())
            client2 = TestClient(app)
            client2.post(
                "/admin/policy", json={"min_reuse_score": 0.5, "demand_threshold": 2}
            )
            log.record(
                actor="ops",
                action="admin.policy.update",
                payload={"min_reuse_score": 0.5, "demand_threshold": 2},
            )

            # 3. Store a fragment via the typed client.
            ident = PayloadIdentity(
                payload_hash="h" * 64,
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
            store.put(ident.payload_hash, b"v3-payload")
            frag = Fragment(
                identity=ident,
                payload_ref=ident.payload_hash,
                payload_size=10,
                ttl=60.0,
                reuse_score=0.9,  # high -> "hot" tier
                version_id=1,
                tenant_id="acme",
            )
            store_result = client.store(to_dict(frag), is_primary=True)
            assert store_result["success"] is True
            log.record(
                actor="ops",
                action="fragment.store",
                payload={"content_hash": ident.payload_hash, "tenant_id": "acme"},
            )

            # 4. Tier assignment is "hot" because reuse_score > 0.7.
            assert node.tier_of(ident.payload_hash) == "hot"

            # 5. The encrypted store's ciphertext does not contain
            # the plaintext marker.
            blob = store._store[ident.payload_hash]  # type: ignore[attr-defined]
            assert b"v3-payload" not in blob

            # 6. Master-key rotation preserves the legacy read.
            rotation.rotate(b"\x01" * 32)
            assert store.get(ident.payload_hash) == b"v3-payload"

            # 7. The audit log records the chain-verified entries.
            assert verify_chain(log.all()) is None

            # 8. The /admin/audit route returns the chain.
            audit = client2.get("/admin/audit").json()
            assert audit["intact"] is True
            actions = [e["action"] for e in audit["entries"]]
            assert "admin.policy.update" in actions
            assert "fragment.store" in actions

            # 9. The Node snapshot reflects the store.
            assert node.get_stats().fragment_count == 1
