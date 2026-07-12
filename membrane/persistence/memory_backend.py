"""InMemoryBackend: zero-dependency persistence fallback.

When Redis is unavailable (or the user opts out), this backend
stores fragments, inventory, and metadata in plain Python
objects.

The backend is a thin convenience layer used by the
:class:`~membrane.fragment_store.FragmentStore` and similar
components to abstract over the *physical* storage. It is
process-local and loses all state when the process exits — use
:class:`~membrane.persistence.redis_backend.RedisBackend` when
durability or cross-process sharing is required.

Thread safety:
    The class is **not thread-safe**. Provide external locking
    when sharing across threads.
"""

import time

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature


class InMemoryBackend:
    """In-memory persistence layer for Membrane fragments.

    This is the default fallback when Redis is not installed or
    not running. All state is lost when the process exits.

    Attributes:
        _fragments: Mapping from ``content_hash`` to the stored
            :class:`Fragment`.
        _node_fragments: Per-node set of hashes that the node
            is responsible for.
        _primary: Mapping from ``content_hash`` to the primary
            node ID (when one has been declared).
        _locations: Mapping from ``content_hash`` to the set of
            node IDs that have reported holding the fragment
            (used by directory lookups).
        _lru: Per-fragment LRU timestamp, used to compute
            eviction candidates.
    """

    def __init__(self) -> None:
        """Initialize all internal dictionaries to empty."""
        self._fragments: dict[str, Fragment] = {}
        self._node_fragments: dict[str, set[str]] = {}
        self._primary: dict[str, str] = {}
        self._locations: dict[str, set[str]] = {}
        self._lru: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fragment CRUD
    # ------------------------------------------------------------------

    def store_fragment(self, fragment: Fragment, node_id: str, is_primary: bool = False) -> bool:
        """Store ``fragment`` under ``node_id``.

        Updates the per-node fragment set, the primary map (when
        ``is_primary`` is True), and the LRU timestamp. The
        method returns ``True`` unconditionally because the
        in-memory backend cannot reject a write.

        Args:
            fragment: Fragment to store.
            node_id: Node identifier owning the fragment.
            is_primary: Whether this node is the primary owner.

        Returns:
            bool: Always ``True`` for the in-memory backend.
        """
        h = fragment.content_hash
        self._fragments[h] = fragment
        self._node_fragments.setdefault(node_id, set()).add(h)
        if is_primary:
            self._primary[h] = node_id
        # Refresh LRU on every write.
        self._lru[h] = time.time()
        return True

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Retrieve a fragment by ``content_hash``.

        Refreshes the LRU timestamp on a hit so frequently
        accessed fragments are protected from eviction.

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment | None: The fragment, or ``None`` if it is
            not stored.
        """
        frag = self._fragments.get(content_hash)
        if frag is not None:
            self._lru[content_hash] = time.time()
        return frag

    def delete_fragment(self, content_hash: str) -> bool:
        """Remove a fragment from every internal table.

        Args:
            content_hash: Hash to delete.

        Returns:
            bool: Always ``True`` for the in-memory backend.
        """
        self._fragments.pop(content_hash, None)
        self._primary.pop(content_hash, None)
        self._lru.pop(content_hash, None)
        # Garbage-collect the fragment from every node's set
        # so stale entries do not linger.
        for node_set in self._node_fragments.values():
            node_set.discard(content_hash)
        return True

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def inventory_digest(self, node_id: str) -> dict[str, int]:
        """Return ``content_hash -> version_id`` for ``node_id``.

        Args:
            node_id: Node identifier.

        Returns:
            dict[str, int]: Snapshot of the node's inventory.
            Empty when the node holds no fragments.
        """
        digest: dict[str, int] = {}
        for h in self._node_fragments.get(node_id, set()):
            frag = self._fragments.get(h)
            if frag is not None:
                digest[h] = frag.version_id
        return digest

    def list_node_fragments(self, node_id: str) -> set[str]:
        """Return the set of content hashes that ``node_id`` holds.

        Args:
            node_id: Node identifier.

        Returns:
            set[str]: Defensive copy of the node's fragment set.
        """
        return set(self._node_fragments.get(node_id, set()))

    # ------------------------------------------------------------------
    # Location / directory
    # ------------------------------------------------------------------

    def record_location(self, content_hash: str, node_id: str) -> None:
        """Record that ``node_id`` holds ``content_hash``.

        Args:
            content_hash: Fragment content hash.
            node_id: Node identifier.
        """
        self._locations.setdefault(content_hash, set()).add(node_id)

    def locate(self, content_hash: str) -> set[str]:
        """Return all node IDs recorded as holders of ``content_hash``.

        Args:
            content_hash: Fragment content hash.

        Returns:
            set[str]: Defensive copy of the holder set.
        """
        return set(self._locations.get(content_hash, set()))

    def get_primary(self, content_hash: str) -> str | None:
        """Return the primary node for ``content_hash``.

        Args:
            content_hash: Fragment content hash.

        Returns:
            str | None: Primary node ID, or ``None`` if no
            primary has been declared.
        """
        return self._primary.get(content_hash)

    # ------------------------------------------------------------------
    # LRU eviction support
    # ------------------------------------------------------------------

    def lru_candidates(self, count: int) -> list[str]:
        """Return up to ``count`` least-recently-used hashes.

        Args:
            count: Maximum number of candidates to return.

        Returns:
            list[str]: Hashes ordered by oldest access first.
        """
        sorted_items = sorted(self._lru.items(), key=lambda kv: kv[1])
        return [h for h, _ in sorted_items[:count]]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Health check.

        Returns:
            bool: Always ``True`` for the in-memory backend.
        """
        return True

    def flush(self) -> None:
        """Drop all stored state.

        After :meth:`flush`, every internal table is empty.
        """
        self._fragments.clear()
        self._node_fragments.clear()
        self._primary.clear()
        self._locations.clear()
        self._lru.clear()

    # ------------------------------------------------------------------
    # Serialization helpers (passthrough — no-op for in-memory)
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_fragment(fragment: Fragment) -> dict[str, str]:
        """Serialize a fragment to a flat string-keyed dict.

        Used by backends that need a serialization layer (e.g.,
        Redis). The in-memory backend never calls this method
        itself; it is provided for parity with
        :class:`RedisBackend`.

        Args:
            fragment: Fragment to serialize.

        Returns:
            dict[str, str]: Flat string-keyed mapping suitable
            for storage in a key-value backend.
        """
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
    def deserialize_fragment(data: dict[str, str]) -> Fragment:
        """Deserialize a fragment from a flat string-keyed dict.

        Args:
            data: Mapping produced by :meth:`serialize_fragment`.

        Returns:
            Fragment: Reconstructed fragment instance.
        """
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
