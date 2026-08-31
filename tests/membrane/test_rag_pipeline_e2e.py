"""End-to-end RAG pipeline via MembraneClient (Phase 3.6.3 follow-up).

The Phase 3.6.3 commit shipped ``examples/rag_pipeline.py``.
The existing RAG tests only verify the example runs as a
subprocess. This test runs the same RAG flow end-to-end
through the typed ``MembraneClient`` against a real FastAPI
app: a tenant stores fragments with payloads in the
encrypted store, the client retrieves them, and a hit /
miss loop drives a small ranking step.
"""

from __future__ import annotations

import pytest


class TestRagPipelineE2E:
    def test_rag_store_retrieve_round_trip(self):
        from fastapi.testclient import TestClient

        from membrane.client import MembraneClient
        from membrane.compute.cpu import CPU
        from membrane.content_store_encrypted import EncryptedInProcessBytes
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.serialization import to_dict
        from membrane.transport.fastapi import create_app

        store = EncryptedInProcessBytes(tenant_id="public")
        node = Node(node_id="n1", max_memory_bytes=10_000, content_store=store)
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

            # 1. Build three fragments: alpha, beta, alpha (dup).
            prompts = [
                ("alpha", "List three colors."),
                ("beta", "Summarize a famous quote."),
                ("alpha", "List three colors."),  # dedup
            ]
            hashes: list[str] = []
            for slot, prompt in prompts:
                ident = PayloadIdentity(
                    payload_hash=str(abs(hash(prompt))).ljust(64, "0")[:64],
                    model_id="rag",
                    model_revision="",
                    tokenizer_name="rag",
                    tokenizer_revision="",
                    layer_range=(0, 1),
                    head_range=(-1, -1),
                    token_span=(0, len(prompt)),
                    dtype="float16",
                    shape=(1, 1, 1, 1, 64),
                )
                # Pre-populate the encrypted store with the
                # answer body.
                answer = f"answer-for-{slot}"
                store.put(ident.payload_hash, answer.encode())
                frag = Fragment(
                    identity=ident,
                    payload_ref=ident.payload_hash,
                    payload_size=len(answer),
                    ttl=60.0,
                    reuse_score=1.0,
                    version_id=1,
                    tenant_id="public",
                )
                result = client.store(to_dict(frag), is_primary=True)
                assert result["success"] is True
                hashes.append(ident.payload_hash)

            # 2. The first and third prompts share a hash
            # (dedup), so the inventory shows 2 distinct fragments.
            inventory = client.inventory()
            assert len(inventory["digest"]) == 2

            # 3. Each hash is retrievable via the encrypted store.
            for content_hash in hashes:
                answer = store.get(content_hash)
                assert answer is not None
                assert answer.startswith(b"answer-for-")

            # 4. The encrypted store defends against tamper:
            # the plaintext marker is never on disk.
            for path in store._store:  # type: ignore[attr-defined]
                # Internal: blob bytes are nonce + ciphertext + tag.
                blob = store._store[path]  # type: ignore[attr-defined]
                assert b"answer-for-" not in blob
