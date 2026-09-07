"""Prefix cache + KV handle (Phase 7).

The vLLM connector (Phase 5) answers
:func:`get_num_new_matched_tokens` by asking the cluster how
many tokens of an incoming request are already cached. The
v1 of this module is the in-process component that backs
that answer: a content-addressed LRU keyed on a SHA-256
``KVHandle`` that fingerprints ``(model_id, token_prefix)``.

The cache is the single-process / single-node primitive. A
distributed deployment composes this cache with a
:class:`MembraneClusterClient` (Phase 5) so the
:class:`MembraneVLLMConnector` can ask the local cache first
and fall back to the cluster. The :func:`lookup` method
finds the longest matching prefix in O(N) over the entries
in the cache; the v1 keeps a fixed capacity and evicts the
oldest entry once the limit is reached.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KVHandle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVHandle:
    """Stable identifier for a cached K/V prefix.

    Equality is by ``handle`` (the SHA-256 digest); the
    ``created_at`` field is metadata for diagnostics.

    Attributes:
        handle: SHA-256 hex digest of ``(model_id, token_ids)``.
        model_id: Model identity the prefix was computed under.
        token_len: Number of tokens the prefix covers.
        created_at: Monotonic timestamp the handle was minted.
    """

    handle: str
    model_id: str
    token_len: int
    created_at: float

    def __eq__(self, other: object) -> bool:
        """Compare by ``handle`` (the identity of the prefix).

        Args:
            other: Another :class:`KVHandle`.

        Returns:
            bool: ``True`` when both handles share the same
            ``handle`` digest.
        """
        if not isinstance(other, KVHandle):
            return NotImplemented
        return self.handle == other.handle

    def __hash__(self) -> int:
        """Hash by ``handle`` for use in dict / set."""
        return hash(self.handle)

    @classmethod
    def for_prefix(cls, model_id: str, token_ids: tuple[int, ...]) -> KVHandle:
        """Mint a handle for ``(model_id, token_ids)``.

        Args:
            model_id: Model identity the prefix was computed under.
            token_ids: Token sequence the prefix covers.

        Returns:
            KVHandle: A new handle.
        """
        digest = hashlib.sha256(
            f"{model_id}\x00{','.join(str(t) for t in token_ids)}".encode()
        ).hexdigest()
        return cls(
            handle=digest,
            model_id=model_id,
            token_len=len(token_ids),
            created_at=time.monotonic(),
        )

    @classmethod
    def for_tokens(cls, model_id: str, tokens: list[int] | tuple[int, ...]) -> KVHandle:
        """Coerce ``tokens`` to a tuple and mint a handle.

        Args:
            model_id: Model identity.
            tokens: Token sequence as list or tuple.

        Returns:
            KVHandle: A new handle.
        """
        return cls.for_prefix(model_id, tuple(tokens))


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixMatch:
    """Outcome of a prefix-cache lookup.

    Attributes:
        handle: The :class:`KVHandle` that covers the matched
            prefix, or ``None`` when no prefix matched.
        token_len: Number of tokens the matched prefix covers.
        is_full: ``True`` when the matched prefix equals the
            requested token sequence.
    """

    handle: KVHandle | None
    token_len: int
    is_full: bool

    @classmethod
    def miss(cls) -> PrefixMatch:
        """Return a miss result.

        Returns:
            PrefixMatch: ``token_len=0``, ``is_full=False``.
        """
        return cls(handle=None, token_len=0, is_full=False)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """Internal record stored in the cache.

    Attributes:
        handle: Stable handle for the entry.
        token_ids: Token sequence the entry covers.
        layer_range: Inclusive ``(start, end)`` of layer indices.
        last_access: Monotonic timestamp of the last access.
    """

    handle: KVHandle
    token_ids: tuple[int, ...]
    layer_range: tuple[int, int]
    last_access: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------


class PrefixCache:
    """In-memory LRU prefix cache.

    The cache is keyed on the SHA-256 digest of
    ``(model_id, token_prefix)``. Two requests for the same
    prefix collide to the same bucket; the cache keeps the
    longest prefix in the bucket so the lookup is exact.

    Attributes:
        capacity: Maximum number of entries the cache holds.
            When the cache is full, the oldest entry is
            evicted.
    """

    def __init__(self, capacity: int = 1024) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of entries. ``0`` disables
                the cache; ``1`` is the smallest useful value.
        """
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._by_handle: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def insert(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
        layer_range: tuple[int, int],
    ) -> KVHandle:
        """Insert a new prefix into the cache.

        Args:
            model_id: Model identity.
            token_ids: Token sequence the prefix covers.
            layer_range: Inclusive ``(start, end)`` of layer
                indices the cached K/V covers.

        Returns:
            KVHandle: Handle for the inserted entry.
        """
        handle = KVHandle.for_prefix(model_id, token_ids)
        with self._lock:
            if handle.handle in self._by_handle:
                entry = self._by_handle.pop(handle.handle)
                entry.last_access = time.monotonic()
                self._by_handle[handle.handle] = entry
                return entry.handle
            entry = _Entry(
                handle=handle,
                token_ids=tuple(token_ids),
                layer_range=layer_range,
            )
            self._by_handle[handle.handle] = entry
            self._evict_if_needed()
        return handle

    def lookup(self, model_id: str, token_ids: tuple[int, ...]) -> PrefixMatch:
        """Return the longest matching prefix.

        Args:
            model_id: Model identity.
            token_ids: Token sequence to look up.

        Returns:
            PrefixMatch: Longest match, or a miss when no
            prefix covers any of the tokens.
        """
        if not token_ids:
            return PrefixMatch.miss()
        with self._lock:
            best: _Entry | None = None
            for entry in self._by_handle.values():
                if entry.handle.model_id != model_id:
                    continue
                prefix_len = self._matching_prefix_len(entry.token_ids, token_ids)
                if prefix_len == 0:
                    continue
                if best is None or prefix_len > best.handle.token_len:
                    best = entry
                    if prefix_len == len(token_ids):
                        break
            if best is None:
                self._misses += 1
                return PrefixMatch.miss()
            best.last_access = time.monotonic()
            self._by_handle.move_to_end(best.handle.handle)
            self._hits += 1
            is_full = best.handle.token_len == len(token_ids)
            return PrefixMatch(handle=best.handle, token_len=best.handle.token_len, is_full=is_full)

    def evict(self, handle: KVHandle) -> bool:
        """Remove ``handle`` from the cache.

        Args:
            handle: The handle to remove.

        Returns:
            bool: ``True`` if the handle was present.
        """
        with self._lock:
            return self._by_handle.pop(handle.handle, None) is not None

    def clear(self) -> None:
        """Drop every entry from the cache."""
        with self._lock:
            self._by_handle.clear()

    def size(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            int: The entry count.
        """
        with self._lock:
            return len(self._by_handle)

    def stats(self) -> dict[str, int]:
        """Return a stats dict with hits, misses, and size.

        Returns:
            dict: ``{"hits": ..., "misses": ..., "size": ...}``.
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._by_handle),
            }

    @staticmethod
    def _matching_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        """Compute the length of the common prefix of ``a`` and ``b``.

        Args:
            a: First token sequence.
            b: Second token sequence.

        Returns:
            int: Number of matching tokens from the start.
        """
        common = 0
        for x, y in zip(a, b, strict=False):
            if x != y:
                break
            common += 1
        return common

    def _evict_if_needed(self) -> None:
        """Evict the oldest entry until size <= capacity."""
        while self.capacity > 0 and len(self._by_handle) > self.capacity:
            self._by_handle.popitem(last=False)
        if self.capacity == 0:
            self._by_handle.clear()


__all__ = [
    "KVHandle",
    "PrefixCache",
    "PrefixMatch",
]
