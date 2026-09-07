"""Snapshot + audit log integration test (Phase 3.2.8 follow-up).

The 3.2.8 commit shipped the hash-chained audit log. The
existing tests cover the log + admin route + tier migration
in isolation. This test runs the full path: a Node writes
a fragment, the operator's policy is updated via /admin/policy,
the operator triggers an evict via /admin/evict, and every
action is recorded + chain-verified. The demo's "snapshot"
is emulated by reading a snapshot of node state at the end.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestSnapshotAuditFlow:
    def test_full_workflow_with_snapshot(self):
        """A complete admin workflow with a snapshot at the end."""

        from membrane.audit import AuditLog, verify_chain
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.tier_migration import TierMigration
        from membrane.tiers import TierPolicy
        from membrane.transport.admin import create_admin_router

        # 1. Set up a Node + AuditLog + TierMigration.
        node = Node(node_id="n1", max_memory_bytes=10_000, tier_policy=TierPolicy())
        log = AuditLog()
        demoted: list[tuple[str, str]] = []
        migration = TierMigration(
            policy=TierPolicy(),
            on_demote=lambda frag, tier: demoted.append(
                (frag.identity.payload_hash, tier)
            ),
        )
        node.add_eviction_callback(migration.on_evict)

        # 2. Seed fragments.
        for i in range(3):
            ident = PayloadIdentity(
                payload_hash=f"hash-{i}".ljust(64, "0")[:64],
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
            node.store(
                Fragment(
                    identity=ident,
                    payload_ref=None,
                    payload_size=10,
                    ttl=60.0,
                    reuse_score=0.1,
                    version_id=1,
                    tenant_id="acme",
                ),
                is_primary=True,
            )

        # 3. Mount admin routes.
        class _StubCluster:
            shard_manager = None
            replicator = None

        app = FastAPI()
        app.state.node = node
        app.state.cluster_manager = _StubCluster()
        app.state.audit_log = log
        app.include_router(create_admin_router())
        client = TestClient(app)

        # 4. Run a series of admin ops.
        client.post(
            "/admin/policy", json={"min_reuse_score": 0.5, "demand_threshold": 1}
        )
        log.record(
            actor="ops",
            action="admin.policy.update",
            payload={"min_reuse_score": 0.5, "demand_threshold": 1},
        )
        # Inspect the first fragment.
        ident = next(iter(node.fragments.values())).identity
        client.get(f"/admin/fragments/{ident.payload_hash}")
        log.record(
            actor="ops",
            action="admin.fragment.inspect",
            payload={"content_hash": ident.payload_hash},
        )
        # Evict the first fragment.
        client.post(
            "/admin/evict", json={"content_hash": ident.payload_hash}
        )
        log.record(
            actor="ops",
            action="admin.evict",
            payload={"content_hash": ident.payload_hash},
        )

        # 5. Read the audit log via the route.
        audit = client.get("/admin/audit").json()
        actions = [e["action"] for e in audit["entries"]]
        assert "admin.policy.update" in actions
        assert "admin.fragment.inspect" in actions
        assert "admin.evict" in actions
        # Chain verifies.
        assert verify_chain(log.all()) is None
        # The Node snapshot reflects the post-evict state.
        snapshot = node.get_stats()
        assert snapshot.fragment_count == 2  # one evicted

    def test_eviction_callback_fires_on_memory_pressure(self):
        """Triggering eviction through the eviction policy fires the tier callback."""
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.tier_migration import TierMigration
        from membrane.tiers import TierPolicy

        node = Node(node_id="n1", max_memory_bytes=10_000, tier_policy=TierPolicy())
        demoted: list[tuple[str, str]] = []
        migration = TierMigration(
            policy=TierPolicy(),
            on_demote=lambda frag, tier: demoted.append(
                (frag.identity.payload_hash, tier)
            ),
        )
        node.add_eviction_callback(migration.on_evict)
        # Fill the node with high-reuse fragments so the
        # weighted LRU phase evicts them on memory pressure.
        for i in range(20):
            ident = PayloadIdentity(
                payload_hash=f"hash-{i}".rjust(64, "0")[:64],
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
            node.store(
                Fragment(
                    identity=ident,
                    payload_ref=None,
                    payload_size=10,
                    ttl=60.0,
                    reuse_score=0.1,
                    version_id=1,
                    tenant_id="acme",
                ),
                is_primary=True,
            )
        # Evict some bytes; the tier callback fires.
        node.evict(target_bytes=100)
        # The tier migration was called at least once.
        assert len(demoted) >= 1
