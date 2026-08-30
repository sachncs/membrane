"""RAG pipeline end-to-end example (Phase 3.6.3).

This script wires the :class:`membrane.client.MembraneClient`
into a small Retrieval-Augmented Generation loop: the user
emits a prompt; the script checks the local Membrane node for
a cached answer to the same prompt; on a hit the script prints
the cached fragment, on a miss it stores the placeholder
and proceeds. The example is intentionally a single file
that runs end-to-end on a fresh ``python -m membrane.demo``
or against any reachable Membrane server.
"""

from __future__ import annotations

import sys

from membrane.client import MembraneClient
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


def _make_fragment(prompt: str) -> Fragment:
    return Fragment(
        identity=PayloadIdentity(
            payload_hash=str(abs(hash(prompt))),
            model_id="rag-demo",
            model_revision="",
            tokenizer_name="rag",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, len(prompt)),
            dtype="float16",
            shape=(1, 1, 1, 8, 64),
        ),
        payload_ref=None,
        payload_size=len(prompt),
        ttl=60.0,
        reuse_score=1.0,
        version_id=1,
        tenant_id="public",
    )


def main(argv: list[str]) -> int:
    base_url = argv[1] if len(argv) > 1 else "http://localhost:8080"
    client = MembraneClient(base_url)
    prompts = ["What colors are in the flag of France?", "Who wrote Hamlet?"]
    for prompt in prompts:
        frag = _make_fragment(prompt)
        result = client.retrieve(frag.identity.payload_hash)
        if result and result.get("found"):
            print(f"hit  | {prompt}")
        else:
            payload = frag.to_wire_dict() if hasattr(frag, "to_wire_dict") else None
            print(f"miss | {prompt}")
            if payload is not None:
                # Defer real body until v3.0.1.
                client.store(frag.to_wire_dict() if hasattr(frag, "to_wire_dict") else {"schema_version": 5, "tenant_id": "public"})
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
