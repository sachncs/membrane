"""FragmentStore: unified content-addressed fragment storage with multi-factor eviction.

Implements a size-bounded, content-addressed store with TTL expiry and
tiered eviction that combines LRU, reuse-score weighting, and value-density
awareness.  Inspired by selective eviction algorithms surveyed in recent
KV-cache optimization literature.

Tiers (updated on every access):
  * **Hot** — accessed within the last ``hot_ttl`` seconds.
  * **Warm** — accessed within the last ``warm_ttl`` seconds.
  * **Cold** — everything else.

Eviction order:
  1. Expired fragments (past TTL).
  2. Cold fragments, lowest ``value_density`` first.
  3. Warm fragments, weighted LRU (``last_access / reuse_score``).
  4. Hot fragments, only if forced by the size/count cap.
"""

import logging
import time
from dataclasses import dataclass, field

from membrane.fragment import Fragment
from membrane.value_density import ValueDensity

logger = logging.getLogger(__name__)


#: Fraction of capacity that triggers proactive eviction before a store.
CAPACITY_PRESSURE_THRESHOLD: float = 0.90


@dataclass
class FragmentStoreMetrics:
    """Runtime metrics for a FragmentStore."""

    stored_count: int = 0
    stored_bytes: int = 0
    max_bytes: int = 0
    max_count: int = 0
    evicted_count: int = 0
    expired_count: int = 0
    hit_count: int = 0
    miss_count: int = 0


class FragmentStore:
    """Content-addressed fragment store with tiered eviction.

    All fragments are keyed by ``content_hash``.  The store enforces
    ``max_bytes`` and ``max_count`` caps.  When a cap would be exceeded,
    eviction is triggered automatically.

    Args:
        max_bytes: Maximum total size in bytes.
        max_count: Maximum number of fragments.
        hot_ttl: Seconds after which a fragment drops from Hot to Warm.
        warm_ttl: Seconds after which a fragment drops from Warm to Cold.
        value_density: Optional ValueDensity calculator for eviction ranking.
    """

    def __init__(
        self,
        max_bytes: int = 1 << 30,
        max_count: int = 10_000,
        hot_ttl: float = 60.0,
        warm_ttl: float = 300.0,
        value_density: ValueDensity | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_count = max_count
        self.hot_ttl = hot_ttl
        self.warm_ttl = warm_ttl
        self.value_density = value_density or ValueDensity()

        self.fragments: dict[str, Fragment] = {}
        self.access_times: dict[str, float] = {}
        self.insertion_times: dict[str, float] = {}
        self.metrics = FragmentStoreMetrics(
            max_bytes=max_bytes, max_count=max_count
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(self, fragment: Fragment) -> bool:
        """Store a fragment.  Evicts if necessary to respect caps.

        Args:
            fragment: Fragment to store.

        Returns:
            True if stored, False if rejected (fragment larger than max_bytes).
        """
        if fragment.size > self.max_bytes:
            logger.warning(
                "Fragment %s size %s exceeds store limit %s",
                fragment.content_hash,
                fragment.size,
                self.max_bytes,
            )
            return False

        now = time.time()
        h = fragment.content_hash

        # If already present, just refresh metadata
        if h in self.fragments:
            self.access_times[h] = now
            return True

        # Ensure room
        while (
            self.metrics.stored_bytes + fragment.size > self.max_bytes
            or self.metrics.stored_count + 1 > self.max_count
        ):
            evicted = self.evict_one(now)
            if not evicted:
                break

        # Final check after eviction
        if self.metrics.stored_bytes + fragment.size > self.max_bytes:
            logger.warning(
                "Could not store %s: insufficient space after eviction",
                h,
            )
            return False

        self.fragments[h] = fragment
        self.access_times[h] = now
        self.insertion_times[h] = now
        self.metrics.stored_count += 1
        self.metrics.stored_bytes += fragment.size
        logger.debug("Stored fragment %s (%s bytes)", h, fragment.size)
        return True

    def get(self, content_hash: str) -> Fragment | None:
        """Retrieve a fragment by content hash, updating access time.

        Evicts the fragment if its TTL has expired.

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment if present and not expired, else None.
        """
        fragment = self.fragments.get(content_hash)
        if fragment is None:
            self.metrics.miss_count += 1
            return None

        now = time.time()
        age = now - self.insertion_times.get(content_hash, now)
        if age > fragment.ttl:
            logger.debug("Evicting expired fragment %s", content_hash)
            self.remove(content_hash)
            self.metrics.expired_count += 1
            self.metrics.miss_count += 1
            return None

        self.access_times[content_hash] = now
        self.metrics.hit_count += 1
        return fragment

    def remove(self, content_hash: str) -> Fragment | None:
        """Remove a fragment unconditionally and return it.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            The removed fragment, or None if not present.
        """
        fragment = self.fragments.pop(content_hash, None)
        if fragment is None:
            return None
        self.access_times.pop(content_hash, None)
        self.insertion_times.pop(content_hash, None)
        self.metrics.stored_count -= 1
        self.metrics.stored_bytes -= fragment.size
        return fragment

    # ------------------------------------------------------------------
    # Tiered eviction
    # ------------------------------------------------------------------

    def evict_one(self, now: float | None = None) -> Fragment | None:
        """Evict a single fragment according to tiered policy.

        Args:
            now: Optional timestamp for deterministic testing.

        Returns:
            The evicted fragment, or None if nothing evictable.
        """
        now = now if now is not None else time.time()

        # Phase 1: expired
        expired = [
            h
            for h, frag in self.fragments.items()
            if now - self.insertion_times.get(h, now) > frag.ttl
        ]
        if expired:
            h = expired[0]
            frag = self.remove(h)
            if frag is not None:
                self.metrics.expired_count += 1
                self.metrics.evicted_count += 1
            return frag

        # Classify candidates by tier
        hot: list[str] = []
        warm: list[str] = []
        cold: list[str] = []
        for h, frag in self.fragments.items():
            last_access = now - self.access_times.get(h, now)
            if last_access <= self.hot_ttl:
                hot.append(h)
            elif last_access <= self.warm_ttl:
                warm.append(h)
            else:
                cold.append(h)

        # Phase 2: cold by value density (lowest first)
        if cold:
            h = min(
                cold,
                key=lambda h: self.value_density.compute(
                    self.fragments[h], []
                ),
            )
            frag = self.remove(h)
            if frag is not None:
                self.metrics.evicted_count += 1
            return frag

        # Phase 3: warm by weighted LRU
        if warm:
            h = min(
                warm,
                key=lambda h: self.access_times.get(h, now)
                / (self.fragments[h].reuse_score + 0.01),
            )
            frag = self.remove(h)
            if frag is not None:
                self.metrics.evicted_count += 1
            return frag

        # Phase 4: hot — only if forced
        if hot:
            h = min(
                hot,
                key=lambda h: self.access_times.get(h, now)
                / (self.fragments[h].reuse_score + 0.01),
            )
            frag = self.remove(h)
            if frag is not None:
                self.metrics.evicted_count += 1
            return frag

        return None

    def evict_to_target(self, target_bytes: int, now: float | None = None) -> list[str]:
        """Evict fragments until at least *target_bytes* are freed.

        Args:
            target_bytes: Bytes to free.
            now: Optional timestamp.

        Returns:
            List of evicted content hashes.
        """
        if target_bytes <= 0:
            return []

        evicted: list[str] = []
        freed = 0
        while freed < target_bytes:
            frag = self.evict_one(now)
            if frag is None:
                break
            freed += frag.size
            evicted.append(frag.content_hash)
        return evicted

    # ------------------------------------------------------------------
    # Tier queries
    # ------------------------------------------------------------------

    def tier(self, content_hash: str, now: float | None = None) -> str:
        """Return the thermal tier of a fragment.

        Args:
            content_hash: Hash to classify.
            now: Optional timestamp.

        Returns:
            One of ``"hot"``, ``"warm"``, ``"cold"``, or ``"missing"``.
        """
        if content_hash not in self.fragments:
            return "missing"
        now = now if now is not None else time.time()
        last_access = now - self.access_times.get(content_hash, now)
        if last_access <= self.hot_ttl:
            return "hot"
        if last_access <= self.warm_ttl:
            return "warm"
        return "cold"

    def tier_counts(self, now: float | None = None) -> dict[str, int]:
        """Return the number of fragments in each tier.

        Args:
            now: Optional timestamp.

        Returns:
            Dict mapping tier name to count.
        """
        now = now if now is not None else time.time()
        counts: dict[str, int] = {"hot": 0, "warm": 0, "cold": 0}
        for h in self.fragments:
            counts[self.tier(h, now)] += 1
        return counts

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> int:
        """Return current stored bytes."""
        return self.metrics.stored_bytes

    def get_hit_rate(self) -> float:
        """Return hit rate between 0.0 and 1.0."""
        total = self.metrics.hit_count + self.metrics.miss_count
        if total == 0:
            return 0.0
        return self.metrics.hit_count / total

    def get_miss_rate(self) -> float:
        """Return miss rate between 0.0 and 1.0."""
        total = self.metrics.hit_count + self.metrics.miss_count
        if total == 0:
            return 0.0
        return self.metrics.miss_count / total

    def values(self) -> list[Fragment]:
        """Return all stored fragments."""
        return list(self.fragments.values())

    def keys(self) -> set[str]:
        """Return all stored content hashes."""
        return set(self.fragments.keys())
