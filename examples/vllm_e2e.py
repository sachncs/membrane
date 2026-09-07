"""vLLM end-to-end example (Phase 3.6.3 follow-up).

The v3.0.0 release ships :class:`membrane.adapters.vllm.MembraneVLLMConnector`
that speaks the vLLM ``KVConnector`` protocol. The
``MembraneVLLMConnector`` is the production path for a real
vLLM cluster; this example runs the same in-process demo
loop against the :class:`membrane.client.MembraneClient` to
show how a hypothetical vLLM caller would use the v3.0.0
client surface end-to-end.

For a real vLLM integration the caller wires the
connector to the vLLM scheduler + model runner. This script
exercises the Membrane side of the same flow.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    base_url = argv[1] if len(argv) > 1 else "http://localhost:8080"
    import httpx
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
    node = Node(node_id="vllm-demo", max_memory_bytes=10_000_000, content_store=store)
    app = create_app(
        node=node,
        compute_backend=CPU(),
        transfer_service=None,
        cluster_manager=None,
    )
    http = TestClient(app)

    with httpx.Client(
        transport=http._transport,  # type: ignore[attr-defined]
        base_url=base_url,
        timeout=5.0,
    ) as shared:
        client = MembraneClient(base_url, transport=shared)
        prompts = [
            "Hello world",
            "What is 2+2?",
            "Hello world",  # cache hit expected
        ]
        for prompt in prompts:
            ident = PayloadIdentity(
                payload_hash=str(abs(hash(prompt))).ljust(64, "0")[:64],
                model_id="vllm-demo",
                model_revision="",
                tokenizer_name="vllm",
                tokenizer_revision="",
                layer_range=(0, 1),
                head_range=(-1, -1),
                token_span=(0, len(prompt)),
                dtype="float16",
                shape=(1, 1, 1, 8, 64),
            )
            frag = Fragment(
                identity=ident,
                payload_ref=ident.payload_hash,
                payload_size=10,
                ttl=60.0,
                reuse_score=1.0,
                version_id=1,
                tenant_id="public",
            )
            store.put(ident.payload_hash, f"completion-for:{prompt}".encode())
            result = client.store(to_dict(frag), is_primary=True)
            print(f"prompt={prompt!r} -> {result!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
