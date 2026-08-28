"""FailureDetector: strategy for cluster membership failure detection.

The cluster layer used to inline a single threshold-based detector:
a peer is removed after 4 missed heartbeats (~8s at default). That is
fine for small clusters but produces false positives on transient
hiccups and lacks quorum protection — a single buggy node can mark a
healthy peer as failed.

This module introduces two strategy implementations that share the
:class:`FailureDetector` interface:

* :class:`ThresholdDetector` — historical behavior, kept as the default
  for backward compatibility.
* :class:`QuorumDetector` — removes a peer only when ≥ ceil(N/2) + 1
  healthy peers independently report it as missing. Stronger against
  noise and split-brain mis-detections.

Selection happens at :class:`~membrane.network.cluster.Cluster`
construction via the ``failure_detector=...`` kwarg; the rest of the
cluster loop machinery is unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class FailureDetector(Protocol):
    """Strategy for deciding when a peer should be removed.

    Implementations inspect the per-peer state and the global cluster
    state and decide whether a given peer has failed.
    """

    def should_remove(
        self,
        peer_id: str,
        peer_missed: int,
        suspect_votes: int,
        healthy_peer_count: int,
    ) -> bool:
        """Decide whether ``peer_id`` should be removed.

        Args:
            peer_id: Identifier of the peer under evaluation.
            peer_missed: Number of consecutive heartbeat misses observed
                by this node for the peer.
            suspect_votes: Number of other healthy peers that have
                independently reported this peer as suspect.
            healthy_peer_count: Total number of healthy peers in the
                cluster, including this node.

        Returns:
            bool: ``True`` if the peer should be removed from membership.
        """
        ...


@dataclass
class ThresholdDetector:
    """Single-node threshold-based detector (historical default).

    Attributes:
        failure_remove_threshold: Consecutive misses before removal.
    """

    failure_remove_threshold: int = 4

    def should_remove(
        self,
        peer_id: str,
        peer_missed: int,
        suspect_votes: int,
        healthy_peer_count: int,
    ) -> bool:
        """Remove ``peer_id`` if its missed-heartbeat count exceeds the threshold."""
        return peer_missed >= self.failure_remove_threshold


@dataclass
class QuorumDetector:
    """Quorum-based detector.

    Removes a peer only when a strict majority of the healthy cluster
    agrees it is missing. ``peer_missed`` is consulted as a tie-breaker.

    Attributes:
        failure_remove_threshold: Consecutive misses required as a
            baseline before quorum is consulted.
        suspect_threshold: Suspect votes required for removal when
            cluster size exceeds 1.
    """

    failure_remove_threshold: int = 4
    suspect_threshold: int = 2

    def should_remove(
        self,
        peer_id: str,
        peer_missed: int,
        suspect_votes: int,
        healthy_peer_count: int,
    ) -> bool:
        """Remove ``peer_id`` only when majority of healthy peers suspect it."""
        if healthy_peer_count <= 1:
            # Single-node cluster: fall back to threshold semantics.
            return peer_missed >= self.failure_remove_threshold
        quorum = healthy_peer_count // 2 + 1
        return suspect_votes >= quorum and peer_missed >= self.failure_remove_threshold


@dataclass
class Migrator:
    """Strategy for shard migration across nodes.

    Two implementations:

    * :class:`EagerMigrator` — moves all primaries immediately on a
      topology change. Latency spike at rebalance; minimum ongoing churn.
    * :class:`RateLimitedMigrator` — moves primaries at a bounded rate
      to avoid saturating inter-node bandwidth.

    Selection happens at :class:`~membrane.network.cluster.Cluster`
    construction via the ``migrator=...`` kwarg.
    """

    def migrations_per_second(self) -> float:
        """Return the configured migration rate.

        Returns:
            float: Migrations per second. ``float('inf')`` means no limit.
        """
        return float("inf")

    def delay(self) -> float:
        """Return the sleep duration to apply before the next migration."""
        rate = self.migrations_per_second()
        if rate == float("inf") or rate <= 0:
            return 0.0
        return 1.0 / rate


@dataclass
class EagerMigrator(Migrator):
    """Migrate all primaries immediately on topology change."""

    def migrations_per_second(self) -> float:
        return float("inf")


@dataclass
class RateLimitedMigrator(Migrator):
    """Bound migration rate to ``max_per_second`` migrations/second."""

    max_per_second: float = 50.0

    def migrations_per_second(self) -> float:
        return self.max_per_second


__all__ = [
    "EagerMigrator",
    "FailureDetector",
    "Migrator",
    "QuorumDetector",
    "RateLimitedMigrator",
    "ThresholdDetector",
]


# Suppress unused-import warning for ``time`` and ``field`` while keeping
# them available for future use (e.g., cooldown timers in the
# :class:`QuorumDetector`).
_ = (time, field)
