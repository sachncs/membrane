"""Cost-aware tiers + online bandit + cost-router wiring (Phase 3.5.6 + 3.5.7 + 3.5.8).

Phase 3.5 wires the v2.0 decision classes into the v3.0
serving path. This module ships:

* :class:`TierPolicy` + :class:`HotTier` / :class:`WarmTier` /
  :class:`ColdTier` / :class:`ArchivalTier` and the
  :func:`select_tier` helper (3.5.6).
* :class:`Bandit` ε-greedy learner that tunes the
  ``EconomicRouterConfig`` weights from observed hit-rate
  reward (3.5.8).
* :func:`record_op_store` + :func:`record_op_retrieve` glue
  the gates (AdmissionPolicy + TenantQuota) into the real op
  paths without inflating the call sites (3.5.7).
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass
from typing import Any

from membrane.decision import AdmissionPolicy, TenantQuota

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3.5.6 Tiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierPolicy:
    """Per-fragment tier selection.

    Attributes:
        hot_threshold: Fragment reuse_score above which the
            admission lands the fragment in the Hot tier.
        warm_threshold: Reuse score above which the tier is
            Warm; below it the tier is Cold.
        archive_threshold: Reuse score above which Cold is
            retained; otherwise the fragment is archival.
        promote_after_hits: Hit count after which a Cold
            fragment is promoted back to Hot.
    """

    hot_threshold: float = 0.7
    warm_threshold: float = 0.4
    archive_threshold: float = 0.0
    promote_after_hits: int = 3


class HotTier:
    """In-process hot tier (the default :class:`FilesystemBlob`)."""

    def name(self) -> str:
        """Return the human-readable tier name.

        Returns:
            str: ``"hot"``.
        """
        return "hot"

    def accept(self, fragment: Any) -> bool:
        """Return True when ``fragment`` is eligible for the hot tier.

        Args:
            fragment: Any object with a ``reuse_score`` field.

        Returns:
            bool: True when the fragment has a high reuse
            score; the :class:`TierPolicy` decides the
            threshold.
        """
        return getattr(fragment, "reuse_score", 0.0) >= 0.7


class WarmTier:
    """Warm tier (e.g., a second FilesystemBlob instance on warm
    spinning disks)."""

    def name(self) -> str:
        """Return the human-readable tier name.

        Returns:
            str: ``"warm"``.
        """
        return "warm"


class ColdTier:
    """Cold tier (e.g., LMCache or an S3-compatible object store)."""

    def __init__(self, storage: Any | None = None) -> None:
        """Initialize the cold tier.

        Args:
            storage: Pluggable cold-storage backend.
        """
        self.storage = storage

    def name(self) -> str:
        """Return the human-readable tier name.

        Returns:
            str: ``"cold"``.
        """
        return "cold"


class ArchivalTier:
    """Archival tier (S3 / Glacier / Azure Archive)."""

    def name(self) -> str:
        """Return the human-readable tier name.

        Returns:
            str: ``"archival"``.
        """
        return "archival"


def select_tier(policy: TierPolicy, fragment: Any) -> str:
    """Pick a tier based on the fragment's reuse score.

    Args:
        policy: The tier policy.
        fragment: Any object with a ``reuse_score`` field.

    Returns:
        str: ``"hot"``, ``"warm"``, ``"cold"``, or ``"archival"``.
    """
    score = float(getattr(fragment, "reuse_score", 0.0))
    if score >= policy.hot_threshold:
        return "hot"
    if score >= policy.warm_threshold:
        return "warm"
    if score >= policy.archive_threshold:
        return "cold"
    return "archival"


# ---------------------------------------------------------------------------
# 3.5.8 Bandit
# ---------------------------------------------------------------------------


@dataclass
class BanditArm:
    """One arm of the bandit.

    Attributes:
        name: Arm identifier (e.g., ``"latency_ms"``).
        weight: Current weight in the ``EconomicRouterConfig``.
        pulls: Total times the arm has been pulled.
        reward_sum: Cumulative reward.
    """

    name: str
    weight: float
    pulls: int = 0
    reward_sum: float = 0.0


class Bandit:
    """ε-greedy multi-armed bandit.

    Attributes:
        epsilon: Exploration rate in (0, 1]. ``1.0`` always
            explores; ``0.05`` explores 5% of the time.
    """

    def __init__(self, arms: list[BanditArm], epsilon: float = 0.05) -> None:
        """Initialize the bandit.

        Args:
            arms: List of arms.
            epsilon: Exploration rate.
        """
        self.arms = list(arms)
        self.epsilon = epsilon
        self._lock = threading.RLock()

    def select_arm(self) -> BanditArm:
        """Return an arm via ε-greedy selection.

        Returns:
            BanditArm: The selected arm.
        """
        with self._lock:
            if random.random() < self.epsilon:
                return random.choice(self.arms)
            return max(self.arms, key=lambda arm: self._estimated_reward(arm))

    def update(self, arm: BanditArm, reward: float) -> None:
        """Record reward for ``arm`` and update estimated reward.

        Args:
            arm: The arm that produced ``reward``.
            reward: Observed reward (e.g., the hit rate observed
                at the routed store / retrieve op).
        """
        with self._lock:
            arm.pulls += 1
            arm.reward_sum += reward
            # Recompute the weight as the running average.
            arm.weight = arm.reward_sum / max(1, arm.pulls)

    def _estimated_reward(self, arm: BanditArm) -> float:
        """Return the running average reward for ``arm``.

        Args:
            arm: The arm to query.

        Returns:
            float: ``arm.reward_sum / max(1, arm.pulls)`` or
            ``0.5`` as a Laplace-smoothed fallback for arms
            without any pulls yet.
        """
        if arm.pulls == 0:
            return 0.5
        return arm.reward_sum / arm.pulls


# ---------------------------------------------------------------------------
# 3.5.7 Cost router wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EconomicRouterConfigWeights:
    """Tunable weights for the :class:`membrane.economic.Economic` cost router.

    Attributes:
        latency_ms: Weight for the latency dimension.
        bandwidth_cost: Weight for the bandwidth dimension.
        gpu_load: Weight for the GPU load dimension.
        memory_pressure: Weight for the memory pressure
            dimension.
    """

    latency_ms: float = 0.25
    bandwidth_cost: float = 0.25
    gpu_load: float = 0.25
    memory_pressure: float = 0.25

    def normalised(self) -> tuple[float, float, float, float]:
        """Return ``(latency_ms, bandwidth_cost, gpu_load, memory_pressure)``.

        Returns:
            tuple[float, float, float, float]: The four weights
            after normalization (positive; sums to 1).
        """
        total = (
            self.latency_ms + self.bandwidth_cost + self.gpu_load + self.memory_pressure
        )
        if total <= 0:
            return (0.25, 0.25, 0.25, 0.25)
        scale = 1.0 / total
        return (
            self.latency_ms * scale,
            self.bandwidth_cost * scale,
            self.gpu_load * scale,
            self.memory_pressure * scale,
        )


def apply_bandit_to_weights(
    bandit: Bandit | None,
    base: EconomicRouterConfigWeights,
) -> EconomicRouterConfigWeights:
    """Overlay bandit weights on the base router config.

    Args:
        bandit: Optional :class:`Bandit` whose arms carry the
            four weight slots.
        base: The static base weights.

    Returns:
        EconomicRouterConfigWeights: The combined weights.
    """
    if bandit is None or len(bandit.arms) < 4:
        return base
    return EconomicRouterConfigWeights(
        latency_ms=max(0.05, bandit.arms[0].weight),
        bandwidth_cost=max(0.05, bandit.arms[1].weight),
        gpu_load=max(0.05, bandit.arms[2].weight),
        memory_pressure=max(0.05, bandit.arms[3].weight),
    )


def record_op_store(
    fragment: Any,
    *,
    policy: AdmissionPolicy | None = None,
    quota: TenantQuota | None = None,
) -> bool:
    """Decide whether ``fragment`` should be stored.

    Args:
        fragment: The candidate fragment.
        policy: Optional :class:`AdmissionPolicy` gate.
        quota: Optional :class:`TenantQuota`.

    Returns:
        bool: True when both gates (if configured) allow the
        store.
    """
    if policy is not None and not policy.should_admit(
        getattr(fragment, "reuse_score", 0.0)
    ):
        return False
    return quota is None or quota.admit(getattr(fragment, "payload_size", 0))


def record_op_retrieve(
    fragment: Any,
    *,
    quota: TenantQuota | None = None,
) -> bool:
    """Decide whether ``fragment`` can be retrieved under the quotas.

    Args:
        fragment: The candidate fragment.
        quota: Optional :class:`TenantQuota`.

    Returns:
        bool: True when the retrieval is allowed.
    """
    return True  # Quotas gate stores, not reads.


__all__ = [
    "ArchivalTier",
    "Bandit",
    "BanditArm",
    "ColdTier",
    "EconomicRouterConfigWeights",
    "HotTier",
    "TierPolicy",
    "WarmTier",
    "apply_bandit_to_weights",
    "record_op_retrieve",
    "record_op_store",
    "select_tier",
]
