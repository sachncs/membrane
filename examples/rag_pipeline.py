"""Prompt-cache loop against a running Membrane node.

A RAG service checks Membrane before paying for a prefill: each prompt
is content-addressed (SHA-256 of model + prompt), looked up with
``GET /retrieve``, and published with ``POST /store`` on a miss. The
script makes two passes over the same prompts, so the first pass
misses and stores and the second pass hits.

Start a node first, then run the example::

    membrane serve --daemon &
    python examples/rag_pipeline.py

For a secured node set ``MEMBRANE_URL`` and ``MEMBRANE_API_KEY`` (a key
with the ``write`` scope).

The fragments here are metadata-only (``payload_ref=None``): the KV
bytes of a real engine reach the node's content store through an engine
adapter such as ``membrane.adapters.vllm``, not through this API.
"""

import hashlib
import logging
import os

from membrane.client import MembraneClient, MembraneClientError, MembraneConnectionError
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.serialization import to_dict

logger = logging.getLogger(__name__)

MODEL_ID = "rag-demo"
PROMPTS = ["What colors are in the flag of France?", "Who wrote Hamlet?"]


def content_hash(prompt: str) -> str:
    """Stable content address for a prompt under :data:`MODEL_ID`."""
    return hashlib.sha256(f"{MODEL_ID}\0{prompt}".encode()).hexdigest()


def make_fragment(prompt: str) -> Fragment:
    """Describe the cached prefill for ``prompt`` as a fragment."""
    return Fragment(
        identity=PayloadIdentity(
            payload_hash=content_hash(prompt),
            model_id=MODEL_ID,
            model_revision="",
            tokenizer_name=MODEL_ID,
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, len(prompt.split())),
            dtype="float16",
            shape=(1, 1, 1, 8, 64),
        ),
        payload_ref=None,
        payload_size=len(prompt),
        ttl=600.0,
        reuse_score=1.0,
        version_id=1,
    )


def main() -> int:
    base_url = os.environ.get("MEMBRANE_URL", "http://localhost:8080")
    client = MembraneClient(base_url, api_key=os.environ.get("MEMBRANE_API_KEY", ""))
    try:
        for attempt in (1, 2):
            logger.info("pass %s", attempt)
            for prompt in PROMPTS:
                result = client.retrieve(content_hash(prompt))
                if result and result.get("found"):
                    logger.info("  hit  | %s", prompt)
                else:
                    client.store(to_dict(make_fragment(prompt)), is_primary=True)
                    logger.info("  miss | %s (stored)", prompt)
    except MembraneConnectionError as exc:
        logger.error("%s\nStart a node first: membrane serve --daemon", exc)
        return 1
    except MembraneClientError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())
