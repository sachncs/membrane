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

The store is the single point of truth for fragment payloads:
the various indexes (exact, semantic, positional, co-access)
track *metadata* and *placement*; this class owns the bytes.
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
    """Runtime metrics for a :class:`FragmentStore`.

    Attributes:
        stored_count: Current number of fragments held in the
            store.
        stored_bytes: Current aggregate payload size in bytes.
        max_bytes: Configured upper bound on ``stored_bytes``.
        max_count: Configured upper bound on ``stored_count``.
        evicted_count: Cumulative count of fragments removed by
            the eviction policy (TTL or pressure).
        expired_count: Cumulative count of fragments removed
            because their TTL elapsed. A subset of
            ``evicted_count``.
        hit_count: Cumulative count of successful lookups.
        miss_count: Cumulative count of failed lookups (including
            those caused by expiry).
    """

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
    ``max_bytes`` and ``max_count`` caps.  When a cap would be
    exceeded, eviction is triggered automatically.

    Args:
        max_bytes: Maximum total size in bytes.
        max_count: Maximum number of fragments.
        hot_ttl: Seconds after which a fragment drops from Hot to
            Warm.
        warm_ttl: Seconds after which a fragment drops from Warm to
            Cold.
        value_density: Optional
            :class:`~membrane.value_density.ValueDensity` calculator
            for eviction ranking. A default is used if omitted.
    """

    def __init__(
        self,
        max_bytes: int = 1 << 30,
        max_count: int = 10_000,
        hot_ttl: float = 60.0,
        warm_ttl: float = 300.0,
        value_density: ValueDensity | None = None,
    ) -> None:
        """Initialize the store with the supplied configuration.

        The metrics dataclass is pre-populated with the configured
        caps so callers can read ``max_bytes`` / ``max_count``
        through :attr:`metrics` even before any insert.
        """
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
        """Store a fragment, evicting as needed to respect the caps.

        Args:
            fragment: Fragment to store.

        Returns:
            bool: True if the fragment is stored (or was already
            present), False if it cannot be stored — either because
            the fragment itself is larger than ``max_bytes`` or
            because eviction could not free enough space.
        """
        if fragment.size > self.max_bytes:
            # A single fragment that is larger than the entire
            # store can never be admitted; reject without mutating
            # state.
            logger.warning(
                "Fragment %s size %s exceeds store limit %s",
                fragment.content_hash,
                fragment.size,
                self.max_bytes,
            )
            return False

        now = time.time()
        h = fragment.content_hash

        # If already present, just refresh metadata — no need to
        # re-insert or trigger eviction.
        if h in self.fragments:
            self.access_times[h] = now
            return True

        # Make room by evicting one fragment at a time until both
        # caps can accommodate the new entry, or until no further
        # eviction is possible.
        while (
            self.metrics.stored_bytes + fragment.size > self.max_bytes
            or self.metrics.stored_count + 1 > self.max_count
        ):
            evicted = self.evict_one(now)
            if not evicted:
                break

        # Final admission check after the eviction loop.
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

        Evicts the fragment (and increments ``expired_count`` /
        ``miss_count``) when its TTL has elapsed.

        Args:
            content_hash: Hash to look up.

        Returns:
            Fragment | None: The fragment if present and not
            expired, otherwise ``None``.
        """
        fragment = self.fragments.get(content_hash)
        if fragment is None:
            self.metrics.miss_count += 1
            return None

        now = time.time()
        age = now - self.insertion_times.get(content_hash, now)
        if age > fragment.ttl:
            # Treat expired entries as misses so that downstream
            # callers don't interpret TTL expiry as a hit.
            logger.debug("Evicting expired fragment %s", content_hash)
            self.remove(content_hash)
            self.metrics.expired_count += 1
            self.metrics.miss_count += 1
            return None

        # Refresh LRU timestamp.
        self.access_times[content_hash] = now
        self.metrics.hit_count += 1
        return fragment

    def remove(self, content_hash: str) -> Fragment | None:
        """Remove a fragment unconditionally and return it.

        Unlike :meth:`evict_one`, this method does *not* count
        against the eviction/expired metrics — it is intended for
        explicit deletion by the caller.

        Args:
            content_hash: Hash of the fragment to remove.

        Returns:
            Fragment | None: The removed fragment, or ``None`` if
            the hash was not present.
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
        """Evict a single fragment according to the tiered policy.

        The phases are: expired first, then cold (by lowest
        ``value_density``), then warm (by weighted LRU), then hot
        (same weighting). The function returns ``None`` if there
        is nothing evictable.

        Args:
            now: Optional timestamp for deterministic testing.
                Defaults to ``time.time()``.

        Returns:
            Fragment | None: The evicted fragment, or ``None`` if
            the store is empty or every fragment is non-evictable.
        """
        now = now if now is not None else time.time()

        # Phase 1: expired fragments (TTL elapsed since insertion).
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

        # Classify candidates by recency tier.
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

        # Phase 2: cold by value density (lowest first).
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

        # Phase 3: warm by weighted LRU.
        # Earlier accesses (smaller access_times) and lower
        # reuse_score both push the candidate earlier in the sort.
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

        # Phase 4: hot — only when forced.
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
        """Evict fragments until at least ``target_bytes`` are freed.

        Calls :meth:`evict_one` repeatedly until the cumulative
        freed space meets or exceeds ``target_bytes`` or no
        further eviction is possible.

        Args:
            target_bytes: Bytes to free. Values ``<= 0`` short
                circuit and return an empty list.
            now: Optional timestamp forwarded to
                :meth:`evict_one`.

        Returns:
            list[str]: Content hashes of the evicted fragments, in
            eviction order.
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
            str: One of ``"hot"``, ``"warm"``, ``"cold"``, or
            ``"missing"`` (when the hash is not in the store).
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
            dict[str, int]: Mapping ``{"hot": int, "warm": int,
            "cold": int}``. Missing fragments are not counted;
            keys are always present, even with zero values.
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
        """Return current stored bytes.

        Returns:
            int: Total payload size in bytes, equal to
            ``metrics.stored_bytes``.
        """
        return self.metrics.stored_bytes

    def get_hit_rate(self) -> float:
        """Return cumulative hit rate.

        Returns:
            float: Hit count divided by (hit + miss) count, in
            ``[0, 1]``. Returns ``0.0`` when no lookups have
            occurred yet.
        """
        total = self.metrics.hit_count + self.metrics.miss_count
        if total == 0:
            return 0.0
        return self.metrics.hit_count / total

    def get_miss_rate(self) -> float:
        """Return cumulative miss rate.

        Returns:
            float: Miss count divided by (hit + miss) count, in
            ``[0, 1]``. Returns ``0.0`` when no lookups have
            occurred yet.
        """
        total = self.metrics.hit_count + self.metrics.miss_count
        if total == 0:
            return 0.0
        return self.metrics.miss_count / total

    def values(self) -> list[Fragment]:
        """Return all stored fragments as a fresh list.

        Returns:
            list[Fragment]: Snapshot of the store's contents.
        """
        return list(self.fragments.values())

    def keys(self) -> set[str]:
        """Return all stored content hashes as a fresh set.

        Returns:
            set[str]: Snapshot of the store's key set.
        """
        return set(self.fragments.keys())
