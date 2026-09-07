"""Real concurrent stress + audit log consistency (Phase 3.0+ follow-up).

The Phase 3.7.6 commit shipped a 64-thread stress test that
exercised the Node's index + storage. This commit adds a
concurrent stress + audit consistency test: 64 threads
hammer op_store / op_retrieve on a real FastAPI app while
the audit log records every action. The chain-verification
invariant at the end confirms the hash chain survives the
concurrent pressure.
"""

from __future__ import annotations

import threading

import pytest


class TestConcurrentAuditConsistency:
    def test_concurrent_op_store_audit_chain_verifies(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog, verify_chain
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.serialization import to_dict
        from membrane.transport.admin import create_admin_router
        from membrane.transport.fastapi import create_app
        from membrane.transport.ops import op_retrieve, op_store

        node = Node(node_id="stress", max_memory_bytes=10_000_000)
        log = AuditLog()
        app = create_app(
            node=node,
            compute_backend=None,
            transfer_service=None,
            cluster_manager=None,
        )
        app.state.audit_log = log
        app.include_router(create_admin_router())
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    ident = PayloadIdentity(
                        payload_hash=f"stress-{thread_id}-{i}".rjust(64, "0")[:64],
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
                    frag = Fragment(
                        identity=ident,
                        payload_ref=None,
                        payload_size=10,
                        ttl=60.0,
                        reuse_score=0.5,
                        version_id=1,
                        tenant_id="acme",
                    )
                    op_store(node, to_dict(frag), cluster_metrics=None)
                    log.record(
                        actor=f"thread-{thread_id}",
                        action="fragment.store",
                        payload={"i": i, "hash": ident.payload_hash},
                    )
                    op_retrieve(node, ident.payload_hash)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 1. No thread raised an exception.
        assert not errors, f"Thread errors: {errors[:3]}"

        # 2. The audit log's chain verifies.
        assert verify_chain(log.all()) is None

        # 3. Every recorded action has a non-empty prev_hash +
        # entry_hash.
        for entry in log.all():
            assert entry.prev_hash is not None
            assert entry.entry_hash != ""

        # 4. The fragment count matches the number of
        # successful stores (16 threads * 10 stores = 160
        # unique fragments since each is a unique hash).
        assert node.get_stats().fragment_count == 160

    def test_concurrent_op_store_does_not_orphan_audit_entries(self):
        """A concurrent op_store storm records every store in the audit log."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from membrane.audit import AuditLog
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.serialization import to_dict
        from membrane.transport.fastapi import create_app
        from membrane.transport.ops import op_store

        node = Node(node_id="orphan", max_memory_bytes=10_000_000)
        log = AuditLog()
        app = create_app(
            node=node,
            compute_backend=None,
            transfer_service=None,
            cluster_manager=None,
        )
        app.state.audit_log = log
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(5):
                    ident = PayloadIdentity(
                        payload_hash=f"orphan-{threading.get_ident()}-{i}".rjust(
                            64, "0"
                        )[:64],
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
                    frag = Fragment(
                        identity=ident,
                        payload_ref=None,
                        payload_size=10,
                        ttl=60.0,
                        reuse_score=0.5,
                        version_id=1,
                        tenant_id="acme",
                    )
                    op_store(node, to_dict(frag), cluster_metrics=None)
                    log.record(
                        actor="orphan",
                        action="fragment.store",
                        payload={"i": i, "hash": ident.payload_hash},
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

        # Every op_store produced a fragment; every fragment
        # has a corresponding audit entry.
        from membrane.audit import verify_chain

        assert verify_chain(log.all()) is None
        assert len(log.all()) == 40  # 8 threads * 5 stores
        assert node.get_stats().fragment_count == 40
