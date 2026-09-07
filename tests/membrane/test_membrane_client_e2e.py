"""End-to-end MembraneClient test against a real running FastAPI app.

The v3.0.0 release ships a typed ``MembraneClient`` (Phase
3.6.1) but the existing tests exercise the client against
an ``httpx.MockTransport`` only. This module covers the
client + the production ``create_app`` path so a contract
change on either side surfaces here immediately.
"""

from __future__ import annotations

import pytest


class TestMembraneClientAgainstRealServer:
    def test_store_then_retrieve(self):
        from fastapi.testclient import TestClient

        from membrane.client import MembraneClient

        # 1. Set up an in-process server with an EncryptedInProcessBytes
        # store (the v3.0.0 default for single-process deployments).
        from membrane.content_store_encrypted import EncryptedInProcessBytes
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.transport.fastapi import create_app

        store = EncryptedInProcessBytes(tenant_id="acme")
        # Pre-populate the store with the bytes the fragment
        # will reference.
        store.put("payload-h", b"acme-data-bytes")

        node = Node(node_id="n1", max_memory_bytes=10_000, content_store=store)
        from membrane.compute.cpu import CPU

        app = create_app(
            node=node,
            compute_backend=CPU(),
            transfer_service=None,
            cluster_manager=None,
        )
        http = TestClient(app)
        # Pass the TestClient's underlying httpx transport so the
        # MembraneClient's internal httpx.Client reuses the same
        # connection pool.
        import httpx

        with httpx.Client(
            base_url="http://n1",
            transport=http._transport,  # type: ignore[attr-defined]
            timeout=5.0,
        ) as shared_client:
            client = MembraneClient("http://n1", transport=shared_client)

            # 2. Build a v5 fragment that references the pre-populated
            # payload.
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
            frag = Fragment(
                identity=ident,
                payload_ref="payload-h",
                payload_size=10,
                ttl=60.0,
                reuse_score=0.5,
                version_id=1,
                tenant_id="acme",
            )
            from membrane.serialization import to_dict

            # 3. Store the fragment via the client.
            result = client.store(to_dict(frag), is_primary=True)
            assert result.get("success") is True

            # 4. Verify the inventory reflects the store.
            inventory = client.inventory()
            assert ident.payload_hash in inventory["digest"]
            assert inventory["digest"][ident.payload_hash] == frag.version_id
