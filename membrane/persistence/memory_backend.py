"""InMemoryBackend: zero-dependency persistence fallback.

When Redis is unavailable (or the user opts out), this backend stores
fragments, inventory, and metadata in plain Python objects.
"""

import time
from typing import Any

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


class InMemoryBackend:
    """In-memory persistence layer for Membrane fragments.

    This is the default fallback when Redis is not installed or not running.
    All state is lost when the process exits.
    """

    def __init__(self) -> None:
        self._fragments: dict[str, Fragment] = {}
        self._node_fragments: dict[str, set[str]] = {}
        self._primary: dict[str, str] = {}
        self._locations: dict[str, set[str]] = {}
        self._lru: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fragment CRUD
    # ------------------------------------------------------------------

    def store_fragment(self, fragment: Fragment, node_id: str, is_primary: bool = False) -> bool:
        h = fragment.content_hash
        self._fragments[h] = fragment
        self._node_fragments.setdefault(node_id, set()).add(h)
        if is_primary:
            self._primary[h] = node_id
        self._lru[h] = time.time()
        return True

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        frag = self._fragments.get(content_hash)
        if frag is not None:
            self._lru[content_hash] = time.time()
        return frag

    def delete_fragment(self, content_hash: str) -> bool:
        self._fragments.pop(content_hash, None)
        self._primary.pop(content_hash, None)
        self._lru.pop(content_hash, None)
        for node_set in self._node_fragments.values():
            node_set.discard(content_hash)
        return True

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def inventory_digest(self, node_id: str) -> dict[str, int]:
        digest: dict[str, int] = {}
        for h in self._node_fragments.get(node_id, set()):
            frag = self._fragments.get(h)
            if frag is not None:
                digest[h] = frag.version_id
        return digest

    def list_node_fragments(self, node_id: str) -> set[str]:
        return set(self._node_fragments.get(node_id, set()))

    # ------------------------------------------------------------------
    # Location / directory
    # ------------------------------------------------------------------

    def record_location(self, content_hash: str, node_id: str) -> None:
        self._locations.setdefault(content_hash, set()).add(node_id)

    def locate(self, content_hash: str) -> set[str]:
        return set(self._locations.get(content_hash, set()))

    def get_primary(self, content_hash: str) -> str | None:
        return self._primary.get(content_hash)

    # ------------------------------------------------------------------
    # LRU eviction support
    # ------------------------------------------------------------------

    def lru_candidates(self, count: int) -> list[str]:
        sorted_items = sorted(self._lru.items(), key=lambda kv: kv[1])
        return [h for h, _ in sorted_items[:count]]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        return True

    def flush(self) -> None:
        self._fragments.clear()
        self._node_fragments.clear()
        self._primary.clear()
        self._locations.clear()
        self._lru.clear()

    # ------------------------------------------------------------------
    # Serialization helpers (passthrough — no-op for in-memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(fragment: Fragment) -> dict[str, str]:
        return {
            "content_hash": fragment.content_hash,
            "embedding": str(list(fragment.embedding)),
            "model_id": fragment.structural_signature.model_id,
            "layer_start": str(fragment.structural_signature.layer_range[0]),
            "layer_end": str(fragment.structural_signature.layer_range[1]),
            "token_start": str(fragment.structural_signature.token_span[0]),
            "token_end": str(fragment.structural_signature.token_span[1]),
            "size": str(fragment.size),
            "ttl": str(fragment.ttl),
            "reuse_score": str(fragment.reuse_score),
            "version_id": str(fragment.version_id),
        }

    @staticmethod
    def _deserialize(data: dict[str, str]) -> Fragment:
        import json
        return Fragment(
            content_hash=data["content_hash"],
            embedding=tuple(json.loads(data["embedding"])),
            structural_signature=StructuralSignature(
                model_id=data["model_id"],
                layer_range=(int(data["layer_start"]), int(data["layer_end"])),
                token_span=(int(data["token_start"]), int(data["token_end"])),
            ),
            size=int(data["size"]),
            ttl=float(data["ttl"]),
            reuse_score=float(data["reuse_score"]),
            version_id=int(data["version_id"]),
        )
