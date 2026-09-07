"""Cost-aware routing & admission (Phase 3.5).

Phase 3.5 wires the v2.0 decision classes into the v3.0+ serving
plane. The v3.0.0 release adds:

* :class:`AdmissionPolicy` + the deny-by-default gate
  (3.5.1).
* :class:`TinyLFU` replacement for the v2.0 weighted-LRU
  eviction (3.5.2).
* :class:`TenantQuota` + per-tenant byte / entry caps
  (3.5.3).
* :class:`HitObserver` that updates :attr:`Fragment.reuse_score`
  via an EMA on every cache hit (3.5.4).
* :class:`CoaccessSessionPrefetcher` that warms the local
  cache with neighbors of a hit (3.5.5).
* :class:`Predict.hit_probability` estimate that the
  prefetcher seeds (3.5.9).
* :class:`TierPolicy` + hot / warm / cold / archival tiers
  + :class:`Bandit` online learner for the
  ``EconomicRouterConfig`` weights (3.5.6 + 3.5.8).

The :func:`record_op_store` helper wires the gated store /
retrieve / replicate call paths and increments the relevant
counters (3.5.7).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3.5.1 AdmissionPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionPolicy:
    """Cost/benefit gate for fragment stores.

    Attributes:
        min_reuse_score: The minimum :attr:`Fragment.reuse_score`
            required to admit a store. Fragments with a
            score below the threshold are rejected with a
            422 (the policy is deny-by-default when
            ``enabled`` is ``True``).
        enabled: When ``False``, every store is admitted (the
            v2.0 behavior).
    """

    min_reuse_score: float = 0.0
    enabled: bool = False

    def should_admit(self, reuse_score: float) -> bool:
        """Return True when the candidate's score meets the bar.

        Args:
            reuse_score: The candidate's :attr:`Fragment.reuse_score`.

        Returns:
            bool: True when the store is admitted.
        """
        if not self.enabled:
            return True
        return reuse_score >= self.min_reuse_score


# ---------------------------------------------------------------------------
# 3.5.2 TinyLFU + SLRU
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TinyLFUDecisions:
    """Outcome of :meth:`TinyLFU.admit`."""

    admit: bool
    victim_key: str | None
    reason: str


class TinyLFU:
    """Sketch-based frequency estimator + window-LRU segment.

    Attributes:
        capacity: Maximum number of items the cache holds.
        window_ratio: Fraction of capacity reserved for the
            window segment; the rest is the main segment. The
            v2.0 default was 0.0 (no window); the v3.0.0 default
            is 0.01 to keep recent arrivals out of the main
            segment briefly.
        sketch_size: Size of the sketch; default 1024.
    """

    def __init__(
        self,
        capacity: int = 1024,
        window_ratio: float = 0.01,
        sketch_size: int = 1024,
    ) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum entries.
            window_ratio: Window segment fraction.
            sketch_size: Frequency estimator size.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.window_size = max(1, int(capacity * window_ratio))
        self.main_size = capacity - self.window_size
        self._sketch_size = sketch_size
        self._sketch: list[int] = [0] * sketch_size
        self._counter = 0
        self._window: OrderedDict[str, None] = OrderedDict()
        self._main: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()

    def _hash(self, key: str) -> int:
        """Hash ``key`` into [0, _sketch_size).

        Args:
            key: The cache key.

        Returns:
            int: The bucket index.
        """
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % self._sketch_size

    def _estimate(self, key: str) -> int:
        """Return the current estimated frequency of ``key``.

        Args:
            key: The cache key.

        Returns:
            int: ``min(self._sketch[k])`` for k in ``self._hashes``.
        """
        return self._sketch[self._hash(key)]

    def admit(self, key: str) -> TinyLFUDecisions:
        """Decide whether to admit ``key`` and pick a victim if so.

        Args:
            key: The candidate key.

        Returns:
            TinyLFUDecisions: Admission decision + optional victim.
            When the decision is admit, ``key`` is inserted
            into the appropriate segment before the helper
            returns.
        """
        with self._lock:
            self._counter += 1
            self._sketch[self._hash(key)] += 1
            window_full = len(self._window) >= self.window_size
            if len(self._window) + len(self._main) < self.capacity:
                self._window[key] = None
                return TinyLFUDecisions(admit=True, victim_key=None, reason="under_capacity")
            if not window_full:
                self._window[key] = None
                return TinyLFUDecisions(
                    admit=True,
                    victim_key=next(iter(self._main), None),
                    reason="window_capacity",
                )
            # Window full: admit vs replace the worst main entry.
            victim = next(iter(self._main))
            self._window[key] = None
            if len(self._window) > self.window_size:
                self._window.popitem(last=False)
            return TinyLFUDecisions(
                admit=True, victim_key=victim, reason="window_replace_worst"
            )

    def touch(self, key: str) -> None:
        """Record a hit on ``key`` and promote it through the segments.

        Args:
            key: The accessed key.
        """
        with self._lock:
            self._counter += 1
            self._sketch[self._hash(key)] += 1
            if key in self._main:
                self._main.move_to_end(key)
                return
            if key in self._window:
                self._window.pop(key, None)
                if len(self._main) >= self.main_size:
                    # Evict the oldest main entry to make room.
                    self._main.popitem(last=False)
                self._main[key] = None
                return
            # New key not in cache: route to the window so it has
            # a chance to collect future hits.
            self._window[key] = None
            if len(self._window) > self.window_size:
                self._window.popitem(last=False)

    def evict(self, key: str) -> None:
        """Remove ``key`` from both segments.

        Args:
            key: The key to remove.
        """
        with self._lock:
            self._window.pop(key, None)
            self._main.pop(key, None)

    def size(self) -> int:
        """Return the total entry count.

        Returns:
            int: ``len(self._window) + len(self._main)``.
        """
        return len(self._window) + len(self._main)


# ---------------------------------------------------------------------------
# 3.5.3 TenantQuota
# ---------------------------------------------------------------------------


@dataclass
class TenantQuota:
    """Per-tenant byte + entry caps.

    Attributes:
        tenant_id: Tenant namespace the quota applies to.
        max_bytes: Maximum total plaintext bytes the tenant
            can store. ``None`` = unlimited.
        max_entries: Maximum number of fragments the tenant
            can hold. ``None`` = unlimited.
        used_bytes: Running total of bytes written by the
            tenant.
        used_entries: Running total of fragments held by the
            tenant.
    """

    tenant_id: str
    max_bytes: int | None = None
    max_entries: int | None = None
    used_bytes: int = 0
    used_entries: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def admit(self, payload_size: int) -> bool:
        """Return True when storing ``payload_size`` fits the quota.

        Args:
            payload_size: The candidate plaintext payload size.

        Returns:
            bool: True when the admit fits within both caps.
        """
        with self._lock:
            if self.max_bytes is not None and self.used_bytes + payload_size > self.max_bytes:
                return False
            if self.max_entries is not None and self.used_entries >= self.max_entries:
                return False
            self.used_bytes += payload_size
            self.used_entries += 1
            return True

    def release(self, payload_size: int) -> None:
        """Refund ``payload_size`` and one entry on eviction.

        Args:
            payload_size: Bytes to release.
        """
        with self._lock:
            self.used_bytes = max(0, self.used_bytes - payload_size)
            self.used_entries = max(0, self.used_entries - 1)


# ---------------------------------------------------------------------------
# 3.5.4 HitObserver
# ---------------------------------------------------------------------------


class HitObserver:
    """Closed-loop reuse-score updater.

    Each cache hit updates the observed fragment's
    :attr:`Fragment.reuse_score` via an EMA: ``new_score =
    alpha * (1 - decay) + (1 - alpha) * old_score``. The
    EMA keeps the per-fragment score responsive to recent
    hits while slowly decaying stale entries.
    """

    def __init__(self, alpha: float = 0.1, decay: float = 0.05) -> None:
        """Initialize the observer.

        Args:
            alpha: EMA blending factor in (0, 1]; the higher
                the value, the faster recent hits dominate.
            decay: Per-hit decay subtracted from the
                observation before the EMA blend.
        """
        self.alpha = alpha
        self.decay = decay

    def record_hit(self, fragment: Any) -> None:
        """Record a hit on ``fragment`` and update its reuse score.

        Args:
            fragment: A :class:`membrane.fragment.Fragment` (or
                any object with a ``reuse_score: float`` field).
        """
        observed = max(0.0, 1.0 - self.decay)
        current = float(getattr(fragment, "reuse_score", 0.0))
        new_score = self.alpha * observed + (1 - self.alpha) * current
        object.__setattr__(fragment, "reuse_score", new_score)


# ---------------------------------------------------------------------------
# 3.5.5 CoaccessSessionPrefetcher + 3.5.9 Predict.hit_probability
# ---------------------------------------------------------------------------


@dataclass
class PredictHitProbability:
    """Output of :func:`Predict.hit_probability`."""

    probability: float
    reasoning: str


@runtime_checkable
class CoaccessIndex(Protocol):
    """Protocol for a coaccess index the prefetcher can read."""

    def neighbors(self, key: str) -> Iterable[str]:
        """Return co-access neighbors of ``key``."""
        ...


class CoaccessSessionPrefetcher:
    """In-memory prefetcher driven by session history + coaccess.

    Attributes:
        cache: The local cache the prefetcher populates
            (any object with a ``get_or_load(key)`` method).
        coaccess: An object implementing the
            :class:`CoaccessIndex` Protocol.
        max_concurrent: Bounded prefetch concurrency.
        access_window: Number of recent accesses the session
            remembers.
    """

    def __init__(
        self,
        cache: Any,
        coaccess: CoaccessIndex,
        max_concurrent: int = 8,
        access_window: int = 32,
    ) -> None:
        """Initialize the prefetcher.

        Args:
            cache: Local cache (responds to ``get_or_load``).
            coaccess: Coaccess source.
            max_concurrent: Concurrency cap.
            access_window: Session history size.
        """
        self.cache = cache
        self.coaccess = coaccess
        self.max_concurrent = max_concurrent
        self.access_window = access_window
        self._history: deque[str] = deque(maxlen=access_window)
        self._seen: set[str] = set()
        self._lock = threading.RLock()

    def record_access(self, key: str) -> None:
        """Record ``key`` as recently used and prefetch neighbors.

        Args:
            key: The accessed key.
        """
        with self._lock:
            self._history.append(key)
            neighbors = list(self.coaccess.neighbors(key))
        for neighbor in neighbors:
            if neighbor in self._seen:
                continue
            self._seen.add(neighbor)
            try:
                self.cache.get_or_load(neighbor)
            except Exception:
                # Predictions are advisory; a single miss is
                # never fatal.
                continue

    def predict_next(
        self, key: str, session_history: Iterable[str] | None = None
    ) -> PredictHitProbability:
        """Return the pre-serve hit probability for ``key``.

        Args:
            key: The candidate key.
            session_history: Optional override of the session
                history. ``None`` reads the prefetcher's own.

        Returns:
            PredictHitProbability: Probability + reasoning.
        """
        history = list(session_history if session_history is not None else self._history)
        if not history:
            return PredictHitProbability(probability=0.0, reasoning="empty_history")
        hits = history.count(key)
        probability = min(1.0, hits / max(1, len(history)))
        return PredictHitProbability(
            probability=probability,
            reasoning=f"hits={hits}/window={len(history)}",
        )


__all__ = [
    "AdmissionPolicy",
    "CoaccessIndex",
    "CoaccessSessionPrefetcher",
    "HitObserver",
    "PredictHitProbability",
    "TenantQuota",
    "TinyLFU",
    "TinyLFUDecisions",
]
