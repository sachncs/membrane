"""FailureDetector and Migrator strategies for the cluster layer.

Two strategy interfaces are exposed:

* :class:`FailureDetector` — decides when a peer should be removed
  from membership (threshold or quorum variants).
* :class:`Migrator` — rebalances primaries after a peer leaves
  (eager or rate-limited variants).

Selection happens at :class:`~membrane.network.cluster.Cluster`
construction via the ``failure_detector=...`` and ``migrator=...``
kwargs; the rest of the cluster loop machinery is unchanged.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

    Attributes:
        transfer_fn: Callable that performs the actual transfer of
            one content hash to a new owner. The signature is
            ``(content_hash, leaving_peer) -> None``; implementations
            are expected to update the shard table and call the
            underlying :class:`TransferService` / :class:`Peer`.
    """

    transfer_fn: Callable[[str, str], None] | None = None

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

    def migrate(
        self,
        hashes: Iterable[str],
        leaving_peer: str,
    ) -> int:
        """Reassign ``hashes`` away from ``leaving_peer``.

        Concrete strategies may impose a rate limit; the default
        (:class:`EagerMigrator`) migrates everything at once.

        Args:
            hashes: Iterable of content hashes currently owned by
                ``leaving_peer``.
            leaving_peer: Identifier of the peer being removed.

        Returns:
            int: Number of hashes migrated.
        """
        hashes_list = list(hashes)
        if self.transfer_fn is None:
            # No-op when no transfer function has been wired in.
            return 0
        migrated = 0
        for h in hashes_list:
            self.transfer_fn(h, leaving_peer)
            migrated += 1
            pause = self.delay()
            if pause > 0:
                time.sleep(pause)
        return migrated


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
