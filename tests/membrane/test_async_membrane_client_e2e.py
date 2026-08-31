"""End-to-end AsyncMembraneClient test against a real running FastAPI app."""

from __future__ import annotations

import asyncio

import httpx
import pytest


class TestAsyncMembraneClientE2E:
    def test_store_inventory_round_trip(self):
        from membrane.client import AsyncMembraneClient
        from membrane.compute.cpu import CPU
        from membrane.content_store_encrypted import EncryptedInProcessBytes
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.serialization import to_dict
        from membrane.transport.fastapi import create_app

        store = EncryptedInProcessBytes(tenant_id="public")
        node = Node(node_id="async-demo", max_memory_bytes=10_000, content_store=store)
        app = create_app(
            node=node,
            compute_backend=CPU(),
            transfer_service=None,
            cluster_manager=None,
        )
        # httpx.ASGITransport is the async-friendly transport for
        # FastAPI apps; httpx will handle the protocol details.
        asgi_transport = httpx.ASGITransport(app=app)

        async def run() -> dict:
            async with httpx.AsyncClient(
                transport=asgi_transport,
                base_url="http://n1",
                timeout=5.0,
            ) as shared:
                client = AsyncMembraneClient("http://n1", client=shared)
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
                store.put(ident.payload_hash, b"async-payload")
                frag = Fragment(
                    identity=ident,
                    payload_ref=ident.payload_hash,
                    payload_size=10,
                    ttl=60.0,
                    reuse_score=0.5,
                    version_id=1,
                    tenant_id="public",
                )
                # 1. Store via the async client.
                store_result = await client.store(
                    to_dict(frag), is_primary=True
                )
                # 2. Inventory via the async client.
                inventory = await client.inventory()
                return {"store": store_result, "inventory": inventory}

        result = asyncio.run(run())
        assert result["store"]["success"] is True
        assert "h" * 64 in result["inventory"]["digest"]
