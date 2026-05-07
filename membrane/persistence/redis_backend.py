"""RedisBackend: production-grade persistence for fragments and metadata.

Uses Redis hash sets, sorted sets, and keys for fragment storage,
inventory tracking, and LRU eviction.
"""

import json
import logging
import time
from typing import Any

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class RedisBackend:
    """Redis-backed persistence layer for Membrane fragments.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        prefix: Key prefix for Membrane data (default ``membrane:``).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "membrane:") -> None:
        import redis

        self.client = redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    # ------------------------------------------------------------------
    # Fragment CRUD
    # ------------------------------------------------------------------

    def store_fragment(self, fragment: Fragment, node_id: str, is_primary: bool = False) -> bool:
        """Serialize and store a fragment in Redis.

        Args:
            fragment: Fragment to persist.
            node_id: Node that owns the fragment.
            is_primary: Whether this node is the primary.

        Returns:
            True if stored.
        """
        h = fragment.content_hash
        data = self._serialize(fragment)
        pipe = self.client.pipeline()
        pipe.hset(self._key(f"frag:{h}"), mapping=data)
        pipe.sadd(self._key(f"node:{node_id}:fragments"), h)
        if is_primary:
            pipe.set(self._key(f"primary:{h}"), node_id)
        # LRU tracking: score = last access time
        pipe.zadd(self._key("lru"), {h: time.time()})
        pipe.execute()
        logger.debug("Stored fragment %s on node %s", h, node_id)
        return True

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Deserialize and return a fragment from Redis.

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment if found, else None.
        """
        data = self.client.hgetall(self._key(f"frag:{content_hash}"))
        if not data:
            return None
        # Update LRU score on read
        self.client.zadd(self._key("lru"), {content_hash: time.time()})
        return self._deserialize(data)

    def delete_fragment(self, content_hash: str) -> bool:
        """Remove a fragment from Redis.

        Args:
            content_hash: Hash to remove.

        Returns:
            True if removed.
        """
        pipe = self.client.pipeline()
        pipe.delete(self._key(f"frag:{content_hash}"))
        pipe.delete(self._key(f"primary:{content_hash}"))
        pipe.zrem(self._key("lru"), content_hash)
        pipe.execute()
        logger.debug("Deleted fragment %s", content_hash)
        return True

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def inventory_digest(self, node_id: str) -> dict[str, int]:
        """Return content_hash -> version_id for a node.

        Args:
            node_id: Node to query.

        Returns:
            Inventory digest.
        """
        hashes = self.client.smembers(self._key(f"node:{node_id}:fragments"))
        digest: dict[str, int] = {}
        for h in hashes:
            vid = self.client.hget(self._key(f"frag:{h}"), "version_id")
            if vid is not None:
                digest[h] = int(vid)
        return digest

    def list_node_fragments(self, node_id: str) -> set[str]:
        """Return all content hashes stored on a node.

        Args:
            node_id: Node to query.

        Returns:
            Set of content hashes.
        """
        return self.client.smembers(self._key(f"node:{node_id}:fragments"))

    # ------------------------------------------------------------------
    # Location / directory
    # ------------------------------------------------------------------

    def record_location(self, content_hash: str, node_id: str) -> None:
        """Record that a fragment is located on a node.

        Args:
            content_hash: Fragment hash.
            node_id: Node identifier.
        """
        self.client.sadd(self._key(f"loc:{content_hash}"), node_id)

    def locate(self, content_hash: str) -> set[str]:
        """Return all nodes holding a fragment.

        Args:
            content_hash: Hash to look up.

        Returns:
            Set of node identifiers.
        """
        return self.client.smembers(self._key(f"loc:{content_hash}"))

    def get_primary(self, content_hash: str) -> str | None:
        """Return the primary node for a fragment.

        Args:
            content_hash: Hash to look up.

        Returns:
            Node identifier, or None if not set.
        """
        return self.client.get(self._key(f"primary:{content_hash}"))

    # ------------------------------------------------------------------
    # LRU eviction support
    # ------------------------------------------------------------------

    def lru_candidates(self, count: int) -> list[str]:
        """Return the *count* least-recently-accessed fragment hashes.

        Args:
            count: Number of candidates to return.

        Returns:
            List of content hashes ordered by LRU (oldest first).
        """
        return self.client.zrange(self._key("lru"), 0, count - 1)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return self.client.ping()
        except Exception:
            return False

    def flush(self) -> None:
        """Clear all Membrane keys (dangerous — use only for testing)."""
        for key in self.client.scan_iter(match=self._key("*")):
            self.client.delete(key)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(fragment: Fragment) -> dict[str, str]:
        return {
            "content_hash": fragment.content_hash,
            "embedding": json.dumps(list(fragment.embedding)),
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
