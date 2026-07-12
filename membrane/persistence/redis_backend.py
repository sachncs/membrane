"""RedisBackend: production-grade persistence for fragments and metadata.

Uses Redis hash sets, sorted sets, and keys for fragment
storage, inventory tracking, and LRU eviction.

Key layout (all prefixed by ``prefix``):

* ``{prefix}frag:{hash}`` — hash containing the serialized
  fragment fields.
* ``{prefix}node:{node_id}:fragments`` — set of hashes held by
  ``node_id``.
* ``{prefix}primary:{hash}`` — string holding the primary node
  ID for ``hash`` (when one has been declared).
* ``{prefix}loc:{hash}`` — set of node IDs recorded as holders
  of ``hash`` (directory layer).
* ``{prefix}lru`` — sorted set scoring each hash by its last
  access timestamp; older scores are returned by
  :meth:`lru_candidates`.

The backend performs writes in a Redis pipeline to keep them
atomic from the client's point of view.

Security:
    * The Redis connection URL should be treated as a secret.
      Pass it via the ``MEMBRANE_REDIS_URL`` environment variable
      or a secrets manager rather than hard-coding it in source.
    * The backend does not perform authentication on the data
      itself; rely on Redis ACLs for access control.
"""

import json
import logging
import time
from typing import Any, cast

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class RedisBackend:
    """Redis-backed persistence layer for Membrane fragments.

    Args:
        redis_url: Redis connection URL
            (e.g., ``redis://localhost:6379/0``).
        prefix: Key prefix for Membrane data
            (default ``membrane:``).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "membrane:",
    ) -> None:
        """Initialize the backend with a Redis client.

        Args:
            redis_url: Redis connection URL.
            prefix: Key prefix prepended to every Membrane key.
        """
        # Local import keeps ``import membrane`` cheap when the
        # ``redis`` package is not installed.
        import redis

        self.client = redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix

        # Expose RedisError so callers can narrow exception handling
        # without re-importing redis themselves.
        self.RedisError = redis.RedisError

    def key_for(self, suffix: str) -> str:
        """Return the prefixed Redis key for ``suffix``.

        Args:
            suffix: Key suffix (without prefix).

        Returns:
            str: Fully qualified key.
        """
        return f"{self.prefix}{suffix}"

    # ------------------------------------------------------------------
    # Fragment CRUD
    # ------------------------------------------------------------------

    def store_fragment(
        self,
        fragment: Fragment,
        node_id: str,
        is_primary: bool = False,
    ) -> bool:
        """Serialize and store a fragment in Redis.

        Args:
            fragment: Fragment to persist.
            node_id: Node that owns the fragment.
            is_primary: Whether this node is the primary.

        Returns:
            bool: True if stored.
        """
        h = fragment.content_hash
        data = self.serialize_fragment(fragment)
        # Use a pipeline so the fragment, the per-node set, the
        # primary key, and the LRU score are written atomically
        # from the client's perspective.
        pipe = self.client.pipeline()
        # redis-py's Mapping type accepts dict[str, str] at runtime
        # but the stub signature uses a Union of bytes/bytearray/etc.
        # The cast to the broader Mapping type satisfies mypy without
        # changing the wire format.
        pipe.hset(self.key_for(f"frag:{h}"), mapping=cast(Any, data))
        pipe.sadd(self.key_for(f"node:{node_id}:fragments"), h)
        if is_primary:
            pipe.set(self.key_for(f"primary:{h}"), node_id)
        # LRU tracking: score = last access time.
        pipe.zadd(self.key_for("lru"), {h: time.time()})
        pipe.execute()
        logger.debug("Stored fragment %s on node %s", h, node_id)
        return True

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Deserialize and return a fragment from Redis.

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment | None: The fragment, or ``None`` if not
            stored.
        """
        data = cast(dict[str, str], self.client.hgetall(self.key_for(f"frag:{content_hash}")))
        if not data:
            return None
        # Refresh the LRU score on a hit so frequently accessed
        # fragments are protected from eviction.
        self.client.zadd(self.key_for("lru"), {content_hash: time.time()})
        return self.deserialize_fragment(data)

    def delete_fragment(self, content_hash: str) -> bool:
        """Remove a fragment from Redis.

        Args:
            content_hash: Hash to remove.

        Returns:
            bool: True if removed.
        """
        pipe = self.client.pipeline()
        pipe.delete(self.key_for(f"frag:{content_hash}"))
        pipe.delete(self.key_for(f"primary:{content_hash}"))
        pipe.zrem(self.key_for("lru"), content_hash)
        pipe.execute()
        logger.debug("Deleted fragment %s", content_hash)
        return True

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def inventory_digest(self, node_id: str) -> dict[str, int]:
        """Return ``content_hash -> version_id`` for ``node_id``.

        Args:
            node_id: Node to query.

        Returns:
            dict[str, int]: Inventory digest. Empty when the
            node holds no fragments.
        """
        hashes = cast(set[str], self.client.smembers(self.key_for(f"node:{node_id}:fragments")))
        digest: dict[str, int] = {}
        for h in hashes:
            vid = cast(str | None, self.client.hget(self.key_for(f"frag:{h}"), "version_id"))
            if vid is not None:
                digest[h] = int(vid)
        return digest

    def list_node_fragments(self, node_id: str) -> set[str]:
        """Return all content hashes stored on a node.

        Args:
            node_id: Node to query.

        Returns:
            set[str]: Set of content hashes.
        """
        return cast(set[str], self.client.smembers(self.key_for(f"node:{node_id}:fragments")))

    # ------------------------------------------------------------------
    # Location / directory
    # ------------------------------------------------------------------

    def record_location(self, content_hash: str, node_id: str) -> None:
        """Record that a fragment is located on a node.

        Args:
            content_hash: Fragment hash.
            node_id: Node identifier.
        """
        self.client.sadd(self.key_for(f"loc:{content_hash}"), node_id)

    def locate(self, content_hash: str) -> set[str]:
        """Return all nodes holding a fragment.

        Args:
            content_hash: Hash to look up.

        Returns:
            set[str]: Set of node identifiers.
        """
        return cast(set[str], self.client.smembers(self.key_for(f"loc:{content_hash}")))

    def get_primary(self, content_hash: str) -> str | None:
        """Return the primary node for a fragment.

        Args:
            content_hash: Hash to look up.

        Returns:
            str | None: Node identifier, or ``None`` if no
            primary has been declared.
        """
        return cast(str | None, self.client.get(self.key_for(f"primary:{content_hash}")))

    # ------------------------------------------------------------------
    # LRU eviction support
    # ------------------------------------------------------------------

    def lru_candidates(self, count: int) -> list[str]:
        """Return the ``count`` least-recently-accessed fragment hashes.

        The Redis sorted set ``lru`` scores each hash by its
        last-access timestamp; ``ZRANGE 0 (count - 1)`` therefore
        returns the oldest hashes.

        Args:
            count: Number of candidates to return.

        Returns:
            list[str]: Hashes ordered by LRU (oldest first).
        """
        return cast(list[str], self.client.zrange(self.key_for("lru"), 0, count - 1))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if Redis is reachable.

        Any exception raised by the underlying client is caught
        and turned into ``False`` so callers can use this method
        as a simple liveness probe.

        Returns:
            bool: True when the Redis ping succeeded.
        """
        try:
            return cast(bool, self.client.ping())
        except (self.RedisError, OSError):
            # Connection refused, timeout, or any other Redis/network
            # failure translates to "not reachable" from the caller's
            # perspective.
            return False

    def flush(self) -> None:
        """Clear all Membrane keys (dangerous — use only for testing).

        Walks every key under ``self.prefix`` and deletes it. Do
        not call this in production — it will wipe every
        Membrane fragment held in Redis.
        """
        for key in self.client.scan_iter(match=self.key_for("*")):
            self.client.delete(key)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_fragment(fragment: Fragment) -> dict[str, str]:
        """Serialize a fragment into a flat string-keyed dict.

        Args:
            fragment: Fragment to serialize.

        Returns:
            dict[str, str]: Mapping suitable for storage in a
            Redis hash.
        """
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
    def deserialize_fragment(data: dict[str, str]) -> Fragment:
        """Deserialize a fragment from a flat string-keyed dict.

        Args:
            data: Mapping produced by :meth:`serialize_fragment`.

        Returns:
            Fragment: Reconstructed fragment instance.
        """
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
