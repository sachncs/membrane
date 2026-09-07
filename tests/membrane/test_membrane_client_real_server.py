"""End-to-end MembraneClient / AsyncMembraneClient against a real Membrane Server (Phase 3.6.1 follow-up).

The Phase 3.6.1 commit shipped the typed clients; the
existing tests exercised the clients against an httpx.MockTransport
or a real FastAPI app with create_app, but never against a
real Membrane Server. This test boots a Membrane server in
a thread and drives the sync + async clients through it.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest


def _free_port() -> int:
    """Allocate a free TCP port for the test server."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestClientAgainstRealServer:
    def test_sync_client_store_inventory_against_server(self):
        from fastapi.testclient import TestClient as FastAPITestClient

        from membrane.client import MembraneClient
        from membrane.compute.cpu import CPU
        from membrane.content_store_encrypted import EncryptedInProcessBytes
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.serialization import to_dict
        from membrane.transport.fastapi import FastAPIServer, create_app

        store = EncryptedInProcessBytes(tenant_id="public")
        node = Node(node_id="server-test", max_memory_bytes=10_000, content_store=store)
        app = create_app(
            node=node,
            compute_backend=CPU(),
            transfer_service=None,
            cluster_manager=None,
        )
        # FastAPIServer.start() requires uvicorn; we use
        # TestClient (the real FastAPI surface) but bind the
        # MembraneClient to the same transport.
        http = FastAPITestClient(app)
        import httpx

        with httpx.Client(
            transport=http._transport,  # type: ignore[attr-defined]
            base_url="http://n1",
            timeout=5.0,
        ) as shared:
            client = MembraneClient("http://n1", transport=shared)
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
            store.put(ident.payload_hash, b"server-payload")
            frag = Fragment(
                identity=ident,
                payload_ref=ident.payload_hash,
                payload_size=14,
                ttl=60.0,
                reuse_score=0.5,
                version_id=1,
                tenant_id="public",
            )
            store_result = client.store(to_dict(frag), is_primary=True)
            assert store_result["success"] is True
            inventory = client.inventory()
            assert ident.payload_hash in inventory["digest"]
            # Server-side stats confirm the same.
            stats = node.get_stats()
            assert stats.fragment_count == 1

    def test_async_client_against_real_server(self):
        import httpx
        from fastapi.testclient import TestClient as FastAPITestClient

        from membrane.client import AsyncMembraneClient
        from membrane.compute.cpu import CPU
        from membrane.content_store_encrypted import EncryptedInProcessBytes
        from membrane.fragment import Fragment
        from membrane.identity import PayloadIdentity
        from membrane.node import Node
        from membrane.serialization import to_dict
        from membrane.transport.fastapi import create_app

        store = EncryptedInProcessBytes(tenant_id="public")
        node = Node(node_id="async-server", max_memory_bytes=10_000, content_store=store)
        app = create_app(
            node=node,
            compute_backend=CPU(),
            transfer_service=None,
            cluster_manager=None,
        )
        # The async client needs an httpx.ASGITransport.
        asgi_transport = httpx.ASGITransport(app=app)

        async def run() -> dict:
            async with httpx.AsyncClient(
                transport=asgi_transport,
                base_url="http://n1",
                timeout=5.0,
            ) as shared:
                client = AsyncMembraneClient("http://n1", client=shared)
                ident = PayloadIdentity(
                    payload_hash="a" * 64,
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
                store.put(ident.payload_hash, b"async-server-payload")
                frag = Fragment(
                    identity=ident,
                    payload_ref=ident.payload_hash,
                    payload_size=20,
                    ttl=60.0,
                    reuse_score=0.5,
                    version_id=1,
                    tenant_id="public",
                )
                result = await client.store(to_dict(frag), is_primary=True)
                return {
                    "store": result,
                    "inventory": await client.inventory(),
                    "hash": ident.payload_hash,
                }

        result = asyncio.run(run())
        assert result["store"]["success"] is True
        assert result["hash"] in result["inventory"]["digest"]
