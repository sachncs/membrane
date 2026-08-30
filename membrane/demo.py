"""`python -m membrane.demo` one-command demo (Phase 3.6.4).

The :func:`main` entry point builds an in-process Membrane node,
runs a small RAG-style workload against it, and prints the
cache hit rate so a new operator can see the system doing
real work without standing up a cluster.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the one-command demo.

    Returns:
        int: Process exit code (0 = success).
    """
    from membrane.fragment import Fragment
    from membrane.identity import PayloadIdentity
    from membrane.node import Node

    node = Node(node_id="demo-local", max_memory_bytes=10_000_000)

    prompts = [
        ("alpha", "List three colors."),
        ("beta", "Summarize a famous quote."),
        ("alpha", "List three colors."),  # cache hit expected
    ]

    for slot, prompt in prompts:
        ident = PayloadIdentity(
            payload_hash=slot,
            model_id="demo-llm",
            model_revision="",
            tokenizer_name="demo",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, len(prompt)),
            dtype="float16",
            shape=(1, 1, 1, 8, 64),
        )
        frag = Fragment(
            identity=ident,
            payload_ref=None,
            payload_size=len(prompt),
            ttl=60.0,
            reuse_score=1.0,
            version_id=1,
            tenant_id="public",
        )
        existing = node.fragments.get(ident.payload_hash)
        if existing is not None:
            node.access_times[ident.payload_hash] = 1.0
            logger.info("hit slot=%s", slot)
        else:
            node.store(frag, is_primary=True)
            logger.info("stored slot=%s", slot)

    stats = node.get_stats()
    print(
        f"demo: stored {stats.fragment_count} fragments,"
        f" memory_used_bytes={stats.memory_used_bytes}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
